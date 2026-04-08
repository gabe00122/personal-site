// scripts/generate-posters.mjs
import { execSync } from 'child_process';
import { readdirSync, existsSync } from 'fs';
import { join, basename, extname, dirname } from 'path';

const dir = './static';
const exts = new Set(['.mp4', '.webm', '.mov']);

for (const file of readdirSync(dir, { recursive: true })) {
	const suffix = extname(file);
	if (!exts.has(suffix)) continue;
	const out = join(dir, dirname(file), basename(file, suffix) + '-poster.webp');

	if (existsSync(out)) continue;
	execSync(`ffmpeg -i "${join(dir, file)}" -frames:v 1 -c:v libwebp -quality 80 -y "${out}"`, {
		stdio: 'ignore'
	});
	console.log(basename(out));
}
