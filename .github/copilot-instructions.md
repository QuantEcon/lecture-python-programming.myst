# Python Programming for Economics and Finance - Lecture Series

This repository contains QuantEcon's Python Programming lecture series built using Jupyter Book. The lectures are written in MyST Markdown format and compiled into HTML, PDF, and Jupyter notebook formats.

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

## Working Effectively

### Environment Setup
- **NEVER CANCEL**: Environment creation takes 3+ minutes. Set timeout to 5+ minutes.
- Create conda environment: `conda env create -f environment.yml`
- Activate environment: `eval "$(conda shell.bash hook)" && conda activate quantecon`
- **CRITICAL**: Always activate the quantecon environment before running any jb commands.

### Building the Site
- **NEVER CANCEL**: HTML builds take 4+ minutes. Set timeout to 8+ minutes.
- Build HTML: `jb build lectures --path-output ./ -n -W --keep-going`
- **NEVER CANCEL**: LaTeX dependency installation takes 20+ minutes. Set timeout to 35+ minutes.
- Install LaTeX dependencies: `sudo apt-get update -qq && sudo apt-get install -y texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-fonts-extra texlive-xetex latexmk xindy dvipng cm-super`
- Build PDF LaTeX: `jb build lectures --builder pdflatex --path-output ./ -n -W --keep-going` -- takes 1 minute. Set timeout to 2+ minutes.
- Build Jupyter notebooks: `jb build lectures --path-output ./ --builder=custom --custom-builder=jupyter -n -W --keep-going` -- takes 1 minute. Set timeout to 2+ minutes.

### Compilation Notes
- HTML build always succeeds but returns exit code 1 due to warnings (jax.quantecon.org unreachable) - this is normal.
- PDF compilation using `latexmk -pdf -silent quantecon-python-programming.tex` in `_build/latex/` produces a PDF but has reference warnings - this is normal.
- Jupyter notebook generation works reliably and is cached for subsequent builds.

### Cleaning
- Clean all builds: `jb clean lectures --all`
- Remove build directory: `rm -rf _build`

## Validation
- **ALWAYS run these validation steps** after making changes to ensure functionality.
- Start local server: `python -m http.server 8000 --directory _build/html`
- Verify site loads: Navigate to `http://localhost:8000` and confirm the table of contents displays correctly.
- **MANUAL VALIDATION REQUIREMENT**: Always test navigation by clicking through at least 3 lecture links to verify content renders properly.
- Check output files exist:
  - HTML: `ls -la _build/html/*.html | head -5`
  - Notebooks: `ls -la _build/jupyter/*.ipynb | head -5` 
  - PDF: `ls -la _build/latex/*.pdf`
- **Always validate image rendering** by checking a lecture with plots (e.g., matplotlib or numpy lectures).

### Linting and Code Quality
- Use pylint for Python code validation: `pylint [file]`
- **Always check MyST syntax** by ensuring the build completes without MyST parsing errors.
- The build process includes automatic syntax validation - failed MyST parsing will stop the build.

## Repository Structure

### Key Directories
- `/lectures/` - Contains all MyST Markdown lecture files
- `/lectures/_config.yml` - Jupyter Book configuration 
- `/lectures/_toc.yml` - Table of contents structure
- `/_build/html/` - Generated HTML website
- `/_build/latex/` - Generated LaTeX/PDF files  
- `/_build/jupyter/` - Generated Jupyter notebooks
- `/.github/workflows/` - CI/CD pipelines

### Important Files
- `environment.yml` - Conda environment specification (Python 3.13 + Anaconda + Jupyter Book stack)
- `README.md` - Repository documentation
- `.gitignore` - Excludes `_build/` and other build artifacts

## Common Tasks

### Adding New Lectures
1. Create new `.md` file in `/lectures/` directory using MyST format
2. Add entry to `/lectures/_toc.yml` in appropriate section
3. Build and validate: Follow full build and validation process above
4. **Always test** the new lecture loads correctly in the web interface

### Modifying Existing Lectures  
1. Edit the `.md` file in `/lectures/` directory
2. Build HTML to test changes: `jb build lectures --path-output ./ -n -W --keep-going`
3. **Always validate** the specific lecture page loads and renders correctly
4. Check for broken internal links or references

### Updating Dependencies
1. Modify `environment.yml` for conda packages
2. Recreate environment: `conda env remove -n quantecon && conda env create -f environment.yml`
3. **NEVER CANCEL**: Full environment recreation takes 3+ minutes
4. **Always test** full build process after dependency changes

## Troubleshooting

### Common Issues
- **"jax.quantecon.org unreachable"**: This warning is expected due to network restrictions. Build continues normally.
- **PDF reference warnings**: LaTeX compilation produces warnings about undefined references but still generates a usable PDF.
- **Exit code 1 with warnings**: HTML builds return non-zero exit codes due to warnings but produce valid output.
- **MyST parsing errors**: Check MyST syntax in affected `.md` files. Common issues: incorrect code block syntax, malformed cross-references.

### Build Failures
- **Clean and rebuild**: `rm -rf _build && jb build lectures --path-output ./ -n -W --keep-going`
- **Check conda environment**: Ensure quantecon environment is active before running jb commands
- **LaTeX issues**: Verify LaTeX dependencies are installed if PDF compilation fails

### Performance Notes
- **First build**: Takes 4+ minutes due to notebook execution and image generation
- **Subsequent builds**: Use Jupyter cache, typically complete in similar time
- **PDF compilation**: Fast (~30 seconds) but may have reference issues
- **Memory usage**: Large builds may require sufficient RAM for image processing

## Expected Timing
Reference these timing expectations when setting appropriate timeouts:

- Environment creation: 3 minutes (set timeout: 5+ minutes)
- LaTeX dependencies: 20+ minutes (set timeout: 35+ minutes)  
- HTML build: 4-5 minutes (set timeout: 8+ minutes)
- PDF LaTeX build: 30 seconds (set timeout: 2+ minutes)
- Jupyter build: 45 seconds (set timeout: 2+ minutes)
- PDF compilation: 30 seconds (set timeout: 2+ minutes)

**NEVER CANCEL long-running operations** - they are expected to take significant time.