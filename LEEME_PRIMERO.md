# Versión con resultados incorporados

Ya se revisó el ZIP que recopilaste de Drive. No necesitas volver a ejecutar
`release_check.ipynb` ni enviar otra recopilación para esta revisión.

Esta copia contiene:

- Código y evidencia conservados de la versión anterior.
- `paper_results/`: 334 archivos de resultados, sin paquetes instalados de terceros.
- `figures/`: las cuatro figuras en PDF y PNG, generadas desde sus tablas.
- `docs/RELEASE_EVIDENCE_REVIEW.md`: qué se comprobó y con qué alcance.
- `scripts/audit_saved_results.py`: comprobación repetible de membresías,
  rankings, perfiles y separación entre particiones usando resultados guardados.

## Si quieres repetir las comprobaciones localmente

Desde la carpeta que contiene `pyproject.toml`:

```bash
python -m pip install -e '.[plots]'
python scripts/audit_saved_results.py --out saved_results_check.json
python scripts/make_paper_figures.py --root paper_results --out figures
```

Estas comprobaciones ya se ejecutaron al preparar esta copia. No entrenan modelos.

## Siguiente paso

Revisar e integrar esta copia en una rama del repositorio GitHub existente.
Después se cita el SHA real del commit. Todavía no se ha publicado ni creado un tag.

Conserva tu carpeta de trabajo original y tus resultados completos en Drive.
El notebook de investigación no se presenta como una ejecución completa desde
cero: conserva el control de balance fallido del análisis geométrico exploratorio.
