
import 'dart:io';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ObasApp());
}

class ObasApp extends StatelessWidget {
  const ObasApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'OBAS V2 Mobile',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF176B50),
        ),
        useMaterial3: true,
      ),
      home: const PredictionPage(),
    );
  }
}

class PredictionPage extends StatefulWidget {
  const PredictionPage({super.key});

  @override
  State<PredictionPage> createState() => _PredictionPageState();
}

class _PredictionPageState extends State<PredictionPage> {
  static const int width = 128;
  static const int height = 128;
  static const int channels = 3;

  static const int expectedValues = width * height * channels;
  static const int expectedBinBytes = expectedValues * 4;

  // =========================================================
  // NUEVO: ESTIMACIÓN AMBIENTAL AUTOMÁTICA
  // =========================================================
  // Estos valores son supuestos para demostración.
  // No están validados para todos los parches del dataset.
  //
  // 20 m/píxel -> 400 m²/píxel -> 0.04 ha/píxel.
  //
  // 50 t/ha representa una densidad de biomasa hipotética.
  // El resultado NO representa biomasa realmente consumida.
  //
  // No se modifica el modelo de inteligencia artificial.
  // =========================================================

  static const double estimatedResolutionMeters = 20.0;

  static const double estimatedBiomassTonnesPerHectare = 50.0;

  static const double estimatedHectaresPerPixel =
      (estimatedResolutionMeters * estimatedResolutionMeters) / 10000.0;

  Interpreter? interpreter;

  String status = 'Modelo pendiente de verificar';
  String inputInfo = '-';
  String outputInfo = '-';
  String resultInfo = '-';

  bool processing = false;

  String? selectedFileName;
  String? selectedFilePath;
  int? selectedFileSize;

  Uint8List? originalImageBytes;
  Uint8List? maskImageBytes;
  Uint8List? overlayImageBytes;

  // =========================================================
  // 1. CARGAR MODELO TFLITE
  // =========================================================

  Future<Interpreter> ensureModelLoaded() async {
    if (interpreter != null) {
      return interpreter!;
    }

    final model = await Interpreter.fromAsset(
      'assets/models/student_baseline_float16.tflite',
    );

    if (!mounted) {
      model.close();
      throw StateError('La pantalla ya no está activa.');
    }

    interpreter = model;

    setState(() {
      status = 'Modelo TFLite cargado correctamente';

      inputInfo =
          '${model.getInputTensor(0).shape} | '
          '${model.getInputTensor(0).type}';

      outputInfo =
          '${model.getOutputTensor(0).shape} | '
          '${model.getOutputTensor(0).type}';
    });

    return model;
  }

  Future<void> verifyModel() async {
    try {
      await ensureModelLoaded();
    } catch (error) {
      if (!mounted) return;

      setState(() {
        status = 'Error al cargar el modelo: $error';
      });
    }
  }

  // =========================================================
  // 2. SELECCIONAR ARCHIVO BIN O NPY
  // Compatible con file_picker 13.3.0
  // =========================================================

  Future<void> selectInputFile() async {
    if (processing) return;

    try {
      final List<PlatformFile> files = await FilePicker.pickFiles(
        type: FileType.custom,
        allowedExtensions: ['bin', 'npy'],
      );

      if (!mounted || files.isEmpty) return;

      final PlatformFile file = files.first;
      final String? path = file.path;

      if (path == null || path.isEmpty) {
        setState(() {
          status = 'No se pudo acceder al archivo seleccionado.';
        });
        return;
      }

      final extension = file.name.split('.').last.toLowerCase();

      if (extension != 'bin' && extension != 'npy') {
        throw const FormatException(
          'Formato no permitido. Selecciona BIN o NPY.',
        );
      }

      final int fileSize = await File(path).length();

      if (!mounted) return;

      setState(() {
        selectedFileName = file.name;
        selectedFilePath = path;
        selectedFileSize = fileSize;

        status = 'Archivo seleccionado correctamente.';
        resultInfo = '-';

        originalImageBytes = null;
        maskImageBytes = null;
        overlayImageBytes = null;
      });
    } catch (error) {
      if (!mounted) return;

      setState(() {
        status = 'Error al seleccionar archivo: $error';
      });
    }
  }

  // =========================================================
  // 3. LEER BIN FLOAT32 HWC
  // =========================================================

  Float32List readBin(Uint8List bytes) {
    if (bytes.length != expectedBinBytes) {
      throw FormatException(
        'Tamaño BIN incorrecto. '
        'Esperado: $expectedBinBytes bytes. '
        'Recibido: ${bytes.length} bytes.',
      );
    }

    final data = ByteData.sublistView(bytes);
    final values = Float32List(expectedValues);

    for (int i = 0; i < expectedValues; i++) {
      final value = data.getFloat32(i * 4, Endian.little);

      if (!value.isFinite) {
        throw const FormatException(
          'El archivo BIN contiene valores no válidos.',
        );
      }

      values[i] = value;
    }

    return values;
  }

  // =========================================================
  // 4. LEER NPY FLOAT32 CHW O HWC
  // =========================================================

  Float32List readNpy(Uint8List bytes) {
    if (bytes.length < 12) {
      throw const FormatException('Archivo NPY demasiado pequeño.');
    }

    const magic = [147, 78, 85, 77, 80, 89];

    for (int i = 0; i < magic.length; i++) {
      if (bytes[i] != magic[i]) {
        throw const FormatException('Cabecera NPY no válida.');
      }
    }

    final majorVersion = bytes[6];
    final data = ByteData.sublistView(bytes);

    late final int headerLength;
    late final int headerStart;

    if (majorVersion == 1) {
      headerLength = data.getUint16(8, Endian.little);
      headerStart = 10;
    } else if (majorVersion == 2 || majorVersion == 3) {
      headerLength = data.getUint32(8, Endian.little);
      headerStart = 12;
    } else {
      throw FormatException(
        'Versión NPY no compatible: $majorVersion',
      );
    }

    if (headerStart + headerLength > bytes.length) {
      throw const FormatException('Cabecera NPY incompleta.');
    }

    final header = String.fromCharCodes(
      bytes.sublist(headerStart, headerStart + headerLength),
    );

    final descriptionMatch = RegExp(
      r"""['"]descr['"]\s*:\s*['"]([^'"]+)['"]""",
    ).firstMatch(header);

    final description = descriptionMatch?.group(1);

    if (description != '<f4' && description != '|f4') {
      throw const FormatException(
        'Solo se admiten archivos NPY float32 '
        'little-endian.',
      );
    }

    final fortranMatch = RegExp(
      r"""['"]fortran_order['"]\s*:\s*(True|False)""",
    ).firstMatch(header);

    if (fortranMatch == null ||
        fortranMatch.group(1) != 'False') {
      throw const FormatException(
        'El archivo NPY debe utilizar orden C.',
      );
    }

    final shapeMatch = RegExp(
      r"""['"]shape['"]\s*:\s*\(([^)]*)\)""",
    ).firstMatch(header);

    if (shapeMatch == null) {
      throw const FormatException(
        'No se pudo identificar la forma del NPY.',
      );
    }

    final shape = shapeMatch
        .group(1)!
        .split(',')
        .map((part) => part.trim())
        .where((part) => part.isNotEmpty)
        .map(int.tryParse)
        .toList();

    if (shape.length != 3 || shape.contains(null)) {
      throw const FormatException(
        'El NPY debe tener tres dimensiones.',
      );
    }

    final dimensions = shape.cast<int>();

    final isCHW =
        dimensions[0] == 3 &&
        dimensions[1] == 128 &&
        dimensions[2] == 128;

    final isHWC =
        dimensions[0] == 128 &&
        dimensions[1] == 128 &&
        dimensions[2] == 3;

    if (!isCHW && !isHWC) {
      throw FormatException(
        'Dimensiones incompatibles: $dimensions. '
        'Se espera (3,128,128) o (128,128,3).',
      );
    }

    final dataStart = headerStart + headerLength;

    if (bytes.length - dataStart != expectedBinBytes) {
      throw const FormatException(
        'Cantidad de datos NPY incorrecta.',
      );
    }

    final values = Float32List(expectedValues);

    if (isHWC) {
      for (int i = 0; i < expectedValues; i++) {
        final value = data.getFloat32(
          dataStart + i * 4,
          Endian.little,
        );

        if (!value.isFinite) {
          throw const FormatException(
            'El NPY contiene valores no válidos.',
          );
        }

        values[i] = value;
      }
    } else {
      // CHW -> HWC.
      // No se modifica la normalización original.

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          for (int c = 0; c < channels; c++) {
            final sourceIndex =
                c * height * width + y * width + x;

            final targetIndex =
                (y * width + x) * channels + c;

            final value = data.getFloat32(
              dataStart + sourceIndex * 4,
              Endian.little,
            );

            if (!value.isFinite) {
              throw const FormatException(
                'El NPY contiene valores no válidos.',
              );
            }

            values[targetIndex] = value;
          }
        }
      }
    }

    return values;
  }

  // =========================================================
  // 5. GENERAR IMAGEN ORIGINAL, MASCARA Y SUPERPOSICION
  // =========================================================

  List<Uint8List> generatePredictionImages(
    Float32List values,
    List<List<List<List<double>>>> output,
  ) {
    final original = img.Image(
      width: width,
      height: height,
    );

    final mask = img.Image(
      width: width,
      height: height,
    );

    final overlay = img.Image(
      width: width,
      height: height,
    );

    for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
        final index = (y * width + x) * channels;

        // Conversión únicamente visual.
        // No afecta los datos enviados al modelo.

        final r =
            (values[index].clamp(0.0, 1.0) * 255).toInt();

        final g =
            (values[index + 1].clamp(0.0, 1.0) * 255).toInt();

        final b =
            (values[index + 2].clamp(0.0, 1.0) * 255).toInt();

        original.setPixelRgb(x, y, r, g, b);

        final probability = output[0][y][x][0];
        final isBurned = probability >= 0.5;

        if (isBurned) {
          mask.setPixelRgb(x, y, 255, 255, 255);

          final overlayR =
              (0.6 * 255 + 0.4 * r).round();

          final overlayG =
              (0.4 * g).round();

          final overlayB =
              (0.4 * b).round();

          overlay.setPixelRgb(
            x,
            y,
            overlayR,
            overlayG,
            overlayB,
          );
        } else {
          mask.setPixelRgb(x, y, 0, 0, 0);
          overlay.setPixelRgb(x, y, r, g, b);
        }
      }
    }

    return [
      Uint8List.fromList(img.encodePng(original)),
      Uint8List.fromList(img.encodePng(mask)),
      Uint8List.fromList(img.encodePng(overlay)),
    ];
  }

  // =========================================================
  // 6. EJECUTAR INFERENCIA CON TFLITE
  // =========================================================

  Future<void> predictValues(
    Float32List values,
    String sourceName,
  ) async {
    if (processing) return;

    setState(() {
      processing = true;
      status = 'Ejecutando predicción...';
      resultInfo = '-';

      originalImageBytes = null;
      maskImageBytes = null;
      overlayImageBytes = null;
    });

    try {
      final totalStopwatch = Stopwatch()..start();

      final model = await ensureModelLoaded();

      final inputShape = model.getInputTensor(0).shape;
      final outputShape = model.getOutputTensor(0).shape;

      if (inputShape.toString() != '[1, 128, 128, 3]' ||
          outputShape.toString() != '[1, 128, 128, 1]') {
        throw StateError(
          'Dimensiones inesperadas del modelo.',
        );
      }

      if (values.length != expectedValues) {
        throw const FormatException(
          'Cantidad de valores incorrecta.',
        );
      }

      // Entrada [1,128,128,3].

      final input = [
        List.generate(
          height,
          (y) => List.generate(
            width,
            (x) => List.generate(
              channels,
              (c) =>
                  values[(y * width + x) * channels + c],
            ),
          ),
        ),
      ];

      // Salida [1,128,128,1].

      final List<List<List<List<double>>>> output = [
        List.generate(
          height,
          (_) => List.generate(
            width,
            (_) => <double>[0.0],
          ),
        ),
      ];

      // Medición de inferencia únicamente.

      final inferenceStopwatch = Stopwatch()..start();

      model.run(input, output);

      inferenceStopwatch.stop();

      // =====================================================
      // CALCULAR PIXELES QUEMADOS
      // =====================================================

      int burnedPixels = 0;

      double minValue = double.infinity;
      double maxValue = double.negativeInfinity;

      for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
          final probability = output[0][y][x][0];

          if (!probability.isFinite) {
            throw StateError(
              'El modelo devolvió un valor no válido.',
            );
          }

          if (probability < minValue) {
            minValue = probability;
          }

          if (probability > maxValue) {
            maxValue = probability;
          }

          if (probability >= 0.5) {
            burnedPixels++;
          }
        }
      }

      const totalPixels = width * height;

      final notBurnedPixels =
          totalPixels - burnedPixels;

      final burnedPercentage =
          burnedPixels / totalPixels * 100;

      // =====================================================
      // NUEVO: CALCULAR HECTÁREAS AUTOMÁTICAMENTE
      // =====================================================

      final double estimatedAffectedHectares =
          burnedPixels * estimatedHectaresPerPixel;

      // =====================================================
      // NUEVO: CALCULAR BIOMASA AUTOMÁTICAMENTE
      // =====================================================

      final double estimatedAffectedBiomass =
          estimatedAffectedHectares *
          estimatedBiomassTonnesPerHectare;

      // =====================================================
      // GENERAR IMAGENES
      // =====================================================

      final generatedImages =
          generatePredictionImages(values, output);

      totalStopwatch.stop();

      if (!mounted) return;

      // =====================================================
      // MOSTRAR RESULTADOS
      // =====================================================

      setState(() {
        status = 'Predicción ejecutada correctamente';

        originalImageBytes = generatedImages[0];
        maskImageBytes = generatedImages[1];
        overlayImageBytes = generatedImages[2];

        resultInfo =
            'Archivo: $sourceName\n\n'
            'Píxeles totales: $totalPixels\n'
            'Píxeles quemados: $burnedPixels\n'
            'Píxeles no quemados: $notBurnedPixels\n'
            'Porcentaje quemado: '
            '${burnedPercentage.toStringAsFixed(2)}%\n\n'
            'Valor mínimo de salida: '
            '${minValue.toStringAsFixed(6)}\n'
            'Valor máximo de salida: '
            '${maxValue.toStringAsFixed(6)}\n\n'

            // NUEVOS RESULTADOS AUTOMÁTICOS
            'Hectáreas afectadas (estimación): '
            '${estimatedAffectedHectares.toStringAsFixed(2)} ha\n'
            'Biomasa presente en área afectada (estimación): '
            '${estimatedAffectedBiomass.toStringAsFixed(2)} toneladas\n\n'

            'Parámetros de estimación:\n'
            'Resolución espacial asumida: '
            '${estimatedResolutionMeters.toStringAsFixed(0)} m/píxel\n'
            'Densidad de biomasa asumida: '
            '${estimatedBiomassTonnesPerHectare.toStringAsFixed(0)} t/ha\n'
            'Nota: valores ilustrativos, no mediciones verificadas.\n\n'

            'Latencia de inferencia: '
            '${inferenceStopwatch.elapsedMilliseconds} ms\n'
            'Tiempo total de procesamiento: '
            '${totalStopwatch.elapsedMilliseconds} ms';
      });
    } catch (error) {
      if (!mounted) return;

      setState(() {
        status = 'Error durante la predicción';
        resultInfo = error.toString();
      });
    } finally {
      if (mounted) {
        setState(() {
          processing = false;
        });
      }
    }
  }

  // =========================================================
  // 7. PREDICCION CON PARCHE DE PRUEBA
  // =========================================================

  Future<void> runPrediction() async {
    if (processing) return;

    try {
      final asset = await rootBundle.load(
        'assets/test_data/obas_parche_01_float32.bin',
      );

      final bytes = asset.buffer.asUint8List(
        asset.offsetInBytes,
        asset.lengthInBytes,
      );

      final values = readBin(bytes);

      if (!mounted) return;

      await predictValues(
        values,
        'prueba_01_OBAS_zone1',
      );
    } catch (error) {
      if (!mounted) return;

      setState(() {
        status = 'Error al cargar parche de prueba';
        resultInfo = error.toString();
      });
    }
  }

  // =========================================================
  // 8. PREDICCION CON ARCHIVO SELECCIONADO
  // =========================================================

  Future<void> runSelectedFilePrediction() async {
    if (processing) return;

    final path = selectedFilePath;
    final name = selectedFileName;

    if (path == null || name == null) {
      setState(() {
        status = 'Primero debes seleccionar un archivo.';
      });
      return;
    }

    try {
      final bytes = await File(path).readAsBytes();

      final extension =
          name.split('.').last.toLowerCase();

      late final Float32List values;

      if (extension == 'bin') {
        values = readBin(bytes);
      } else if (extension == 'npy') {
        values = readNpy(bytes);
      } else {
        throw const FormatException(
          'Formato de archivo no compatible.',
        );
      }

      if (!mounted) return;

      await predictValues(values, name);
    } catch (error) {
      if (!mounted) return;

      setState(() {
        status = 'No se pudo procesar el archivo';
        resultInfo = error.toString();
      });
    }
  }

  // =========================================================
  // 9. LIBERAR RECURSOS
  // =========================================================

  @override
  void dispose() {
    interpreter?.close();
    super.dispose();
  }

  // =========================================================
  // 10. MOSTRAR IMAGENES
  // =========================================================

  Widget buildImageSection(
    String title,
    Uint8List? imageBytes,
  ) {
    if (imageBytes == null) {
      return const SizedBox.shrink();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const SizedBox(height: 22),

        Text(
          title,
          style: const TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
          ),
        ),

        const SizedBox(height: 10),

        Container(
          padding: const EdgeInsets.all(10),
          decoration: BoxDecoration(
            color: Colors.grey.shade100,
            borderRadius: BorderRadius.circular(12),
          ),
          child: Image.memory(
            imageBytes,
            height: 250,
            fit: BoxFit.contain,
            filterQuality: FilterQuality.none,
          ),
        ),
      ],
    );
  }

  // =========================================================
  // 11. INTERFAZ PRINCIPAL
  // =========================================================

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('OBAS V2 MOBILE'),
        centerTitle: true,
      ),

      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),

        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,

          children: [
            const Icon(
              Icons.satellite_alt,
              size: 70,
              color: Color(0xFF176B50),
            ),

            const SizedBox(height: 20),

            const Text(
              'Detección de áreas quemadas',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 22,
                fontWeight: FontWeight.bold,
              ),
            ),

            const SizedBox(height: 12),

            const Text(
              'Modelo Student Baseline Float16',
              textAlign: TextAlign.center,
            ),

            const SizedBox(height: 25),

            // VERIFICAR MODELO
            ElevatedButton.icon(
              onPressed:
                  processing ? null : verifyModel,
              icon: const Icon(Icons.memory),
              label: const Text(
                'Verificar modelo de IA',
              ),
            ),

            const SizedBox(height: 12),

            // SELECCIONAR ARCHIVO
            ElevatedButton.icon(
              onPressed:
                  processing ? null : selectInputFile,
              icon: const Icon(Icons.folder_open),
              label: const Text(
                'Seleccionar archivo',
              ),
            ),

            const SizedBox(height: 10),

            Text(
              selectedFileName == null
                  ? 'Ningún archivo seleccionado'
                  : 'Archivo: $selectedFileName',
            ),

            if (selectedFileSize != null)
              Text(
                'Tamaño: $selectedFileSize bytes',
              ),

            const SizedBox(height: 12),

            // PREDICCION DEL ARCHIVO
            ElevatedButton.icon(
              onPressed:
                  processing || selectedFilePath == null
                      ? null
                      : runSelectedFilePrediction,
              icon: const Icon(
                Icons.analytics_outlined,
              ),
              label: const Text(
                'Ejecutar predicción del archivo',
              ),
            ),

            const SizedBox(height: 12),

            // PREDICCION DE PRUEBA
            OutlinedButton.icon(
              onPressed:
                  processing ? null : runPrediction,
              icon: const Icon(
                Icons.science_outlined,
              ),
              label: const Text(
                'Ejecutar predicción de prueba',
              ),
            ),

            const SizedBox(height: 25),

            if (processing)
              const Center(
                child: CircularProgressIndicator(),
              ),

            const SizedBox(height: 15),

            Text(
              'Estado: $status',
              style: const TextStyle(
                fontWeight: FontWeight.bold,
              ),
            ),

            const SizedBox(height: 15),

            Text('Entrada: $inputInfo'),

            const SizedBox(height: 10),

            Text('Salida: $outputInfo'),

            const Divider(height: 35),

            const Text(
              'Resultado de segmentación',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),

            const SizedBox(height: 15),

            SelectableText(resultInfo),

            buildImageSection(
              'Imagen original',
              originalImageBytes,
            ),

            buildImageSection(
              'Máscara de áreas quemadas',
              maskImageBytes,
            ),

            buildImageSection(
              'Áreas quemadas resaltadas',
              overlayImageBytes,
            ),

            const SizedBox(height: 30),
          ],
        ),
      ),
    );
  }
}
