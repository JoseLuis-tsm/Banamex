# Sistema de Inteligencia Crediticia (Tarjetas de Crédito)

Plataforma local que estima, por perfil de adquisición, la línea promedio, el monto total, la tasa de impago y la estrategia sugerida. El contexto completo está en [CLAUDE.md](CLAUDE.md) y el avance en [PLAN.md](PLAN.md).

## Cómo empezar

Se requiere Python 3.10 o superior (el código es compatible con 3.10). Los comandos son para Windows (PowerShell); en macOS/Linux se indica la variante.

1. **Clonar y entrar al repo:**
   `git clone https://github.com/JoseLuis-tsm/Banamex.git` y luego `cd Banamex`
2. **Crear el entorno virtual:** `python -m venv .venv`
3. **Activarlo:** `.venv\Scripts\Activate.ps1` (macOS/Linux: `source .venv/bin/activate`)
4. **Instalar las versiones fijadas:** `python -m pip install -r requirements/lock.txt`
5. **Copiar el CSV de datos** en `data/raw/base_cards_sintetica.csv` (no viene en git), comprobar con `python -m pytest` y abrir los notebooks de `notebooks/` en VS Code o Jupyter con el kernel de `.venv`.

Si PowerShell bloquea la activación, ejecuta una vez `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
