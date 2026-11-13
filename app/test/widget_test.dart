
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:obas_v2_mobile/main.dart';

void main() {
  testWidgets('OBAS V2 MOBILE inicia correctamente', (
    WidgetTester tester,
  ) async {
    await tester.pumpWidget(const ObasApp());

    expect(
      find.text('OBAS V2 MOBILE'),
      findsOneWidget,
    );

    expect(
      find.text('Verificar modelo de IA'),
      findsOneWidget,
    );

    expect(
      find.text('Ejecutar predicción de prueba'),
      findsOneWidget,
    );

    expect(
      find.byType(MaterialApp),
      findsOneWidget,
    );
  });
}