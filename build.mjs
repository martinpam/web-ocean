// Bundles index.html + js modules into self-contained pages that can be opened directly
// from disk (file://) without a server:
//   ocean.html      readable bundle
//   ocean.min.html  minified bundle (WGSL comments/whitespace stripped; JS minified with
//                   esbuild when it can be found, otherwise comments/indentation stripped)
//   node build.mjs
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const root = path.dirname(fileURLToPath(import.meta.url));
const wgslModules = ['js/wgsl/common.js', 'js/wgsl/sky.js', 'js/wgsl/ship.js', 'js/wgsl/sim.js', 'js/wgsl/ocean.js', 'js/wgsl/post.js'];
const order = [...wgslModules, 'js/ship.js', 'js/main.js'];
const stripModuleSyntax = (src) => src
  .replace(/^import[\s\S]*?from\s+['"][^'"]+['"];\s*$/gm, '')                 // drop imports
  .replace(/^export\s+(const|function|async function|let|class)\s/gm, '$1 ');

const page = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
const SCRIPT_TAG = '<script type="module" src="js/main.js"></script>';
const write = (name, html) => {
  fs.writeFileSync(path.join(root, name), html);
  console.log(`wrote ${name}`, (html.length / 1024).toFixed(0) + ' KB');
};

// ---- readable bundle --------------------------------------------------------------------
// the ship mesh travels inside the page as base64 (js/ship.js picks it up)
const shipB64 = fs.readFileSync(path.join(root, 'assets/ship.bin')).toString('base64');
const shipConst = `const SHIP_B64 = '${shipB64}';\n`;
let code = shipConst;
for (const f of order) code += `// ---- ${f} ----\n${stripModuleSyntax(fs.readFileSync(path.join(root, f), 'utf8'))}\n`;
write('ocean.html', page.replace(SCRIPT_TAG, () => `<script type="module">\n${code}\n</script>`));

// ---- minified bundle --------------------------------------------------------------------
// WGSL has no string literals, so comments and whitespace can be removed safely. Spaces are
// kept around + - / so that e.g. "a - -b" or "x / *p" never fuse into other tokens.
function minifyWGSL(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/\/\/[^\n]*/g, ' ')
    .replace(/\s+/g, ' ')
    .replace(/ ?([{}()[\],;:=<>*&|^!%@.]) ?/g, '$1')
    .trim();
}

// Minify every WGSL template literal in the shader modules in place (shared snippets stay
// shared). ${...} interpolations are swapped for identifier-like placeholders first so the
// minifier never fuses tokens across them.
function minifyTemplates(src) {
  return src.replace(/`([^`]*)`/g, (_, body) => {
    const parts = [];
    const masked = body.replace(/\$\{[^}]*\}/g, (m) => `__P${parts.push(m) - 1}__`);
    const out = minifyWGSL(masked).replace(/__P(\d+)__/g, (_, i) => parts[Number(i)]);
    return '`' + out + '`';
  });
}
let minJs = '';
for (const f of wgslModules) minJs += stripModuleSyntax(minifyTemplates(fs.readFileSync(path.join(root, f), 'utf8'))) + '\n';
minJs += stripModuleSyntax(fs.readFileSync(path.join(root, 'js/ship.js'), 'utf8')) + '\n';
minJs += stripModuleSyntax(fs.readFileSync(path.join(root, 'js/main.js'), 'utf8'));

function findEsbuild() {
  const req = createRequire(import.meta.url);
  try { return req('esbuild'); } catch { /* not installed in the project */ }
  // fall back to an esbuild left in the npx cache by other tools
  const cache = path.join(os.homedir(), 'AppData', 'Local', 'npm-cache', '_npx');
  for (const dir of fs.existsSync(cache) ? fs.readdirSync(cache) : []) {
    const pkg = path.join(cache, dir, 'node_modules', 'esbuild');
    if (fs.existsSync(pkg)) { try { return req(pkg); } catch { /* try next */ } }
  }
  return null;
}

const esbuild = findEsbuild();
if (esbuild) {
  minJs = esbuild.transformSync(minJs, { minify: true, format: 'esm', target: 'es2022', legalComments: 'none' }).code;
} else {
  console.log('esbuild not found: JS gets comment/indentation stripping only');
  minJs = minJs.split('\n').map((l) => l.trim()).filter((l) => l && !l.startsWith('//')).join('\n');
}

const minPage = page
  .replace(SCRIPT_TAG, () => `<script type="module">${shipConst}${minJs}</script>`)
  .replace(/<style>([\s\S]*?)<\/style>/, (_, css) => `<style>${css.replace(/\s+/g, ' ').replace(/ ?([{}:;,>]) ?/g, '$1').trim()}</style>`)
  .replace(/>\s+</g, '><');
write('ocean.min.html', minPage);
