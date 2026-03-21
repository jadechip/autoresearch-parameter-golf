#!/usr/bin/env node

import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import process from "node:process";

const DEFAULTS = {
  owner: "openai",
  repo: "parameter-golf",
  state: "all",
  outDir: "artifacts/openai_parameter_golf_prs",
  concurrency: 4,
  maxPrs: null,
};

const USER_AGENT = "codex-github-pr-dataset/1.0";
const GITHUB_API_ACCEPT = "application/vnd.github+json";

function parseArgs(argv) {
  const options = { ...DEFAULTS };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--owner") {
      options.owner = argv[++i];
    } else if (arg === "--repo") {
      options.repo = argv[++i];
    } else if (arg === "--state") {
      options.state = argv[++i];
    } else if (arg === "--out-dir") {
      options.outDir = argv[++i];
    } else if (arg === "--concurrency") {
      options.concurrency = Number.parseInt(argv[++i], 10);
    } else if (arg === "--max-prs") {
      options.maxPrs = Number.parseInt(argv[++i], 10);
    } else if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }
  if (!["open", "closed", "all"].includes(options.state)) {
    throw new Error(`Unsupported state: ${options.state}`);
  }
  if (!Number.isFinite(options.concurrency) || options.concurrency < 1) {
    throw new Error(`Invalid concurrency: ${options.concurrency}`);
  }
  return options;
}

function printHelp() {
  console.log(
    [
      "Usage: node scripts/fetch_github_pr_dataset.mjs [options]",
      "",
      "Options:",
      "  --owner <owner>           GitHub owner (default: openai)",
      "  --repo <repo>             GitHub repo (default: parameter-golf)",
      "  --state <state>           open | closed | all (default: all)",
      "  --out-dir <path>          Output directory",
      "  --concurrency <n>         Parallel PR fetches (default: 4)",
      "  --max-prs <n>             Limit PRs for testing",
      "  --help                    Show this message",
    ].join("\n"),
  );
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function encodeGitHubPath(filePath) {
  return filePath.split("/").map(encodeURIComponent).join("/");
}

function decodeHtmlEntities(text) {
  const named = {
    amp: "&",
    lt: "<",
    gt: ">",
    quot: '"',
    apos: "'",
    nbsp: " ",
  };
  return text.replace(/&(#x?[0-9a-fA-F]+|[a-zA-Z]+);/g, (_, entity) => {
    if (entity.startsWith("#x") || entity.startsWith("#X")) {
      return String.fromCodePoint(Number.parseInt(entity.slice(2), 16));
    }
    if (entity.startsWith("#")) {
      return String.fromCodePoint(Number.parseInt(entity.slice(1), 10));
    }
    return named[entity] ?? `&${entity};`;
  });
}

function htmlInnerText(html) {
  return decodeHtmlEntities(
    html
      .replace(/<!--[\s\S]*?-->/g, "")
      .replace(/<br\s*\/?>/gi, "")
      .replace(/<[^>]+>/g, ""),
  );
}

function csvEscape(value) {
  const stringValue = value == null ? "" : String(value);
  if (/[",\n]/.test(stringValue)) {
    return `"${stringValue.replace(/"/g, '""')}"`;
  }
  return stringValue;
}

async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

function safeResolve(baseDir, relativePath) {
  const resolvedBase = path.resolve(baseDir);
  const resolvedPath = path.resolve(resolvedBase, relativePath);
  if (resolvedPath !== resolvedBase && !resolvedPath.startsWith(`${resolvedBase}${path.sep}`)) {
    throw new Error(`Unsafe path outside base directory: ${relativePath}`);
  }
  return resolvedPath;
}

async function writeText(filePath, text) {
  await ensureDir(path.dirname(filePath));
  await fs.writeFile(filePath, text, "utf8");
}

async function writeBuffer(filePath, buffer) {
  await ensureDir(path.dirname(filePath));
  await fs.writeFile(filePath, buffer);
}

async function fetchWithRetry(url, options = {}) {
  const {
    accept = "*/*",
    retries = 5,
    retryBaseMs = 800,
    headers = {},
  } = options;
  let lastError = null;
  for (let attempt = 0; attempt <= retries; attempt += 1) {
    try {
      const response = await fetch(url, {
        headers: {
          Accept: accept,
          "User-Agent": USER_AGENT,
          ...headers,
        },
      });
      if (response.ok) {
        return response;
      }
      if ([403, 408, 429, 500, 502, 503, 504].includes(response.status) && attempt < retries) {
        const retryAfter = response.headers.get("retry-after");
        const waitMs = retryAfter ? Number.parseInt(retryAfter, 10) * 1000 : retryBaseMs * (attempt + 1);
        await sleep(waitMs);
        continue;
      }
      const body = await response.text();
      throw new Error(`HTTP ${response.status} for ${url}\n${body.slice(0, 400)}`);
    } catch (error) {
      lastError = error;
      if (attempt >= retries) {
        break;
      }
      await sleep(retryBaseMs * (attempt + 1));
    }
  }
  throw lastError ?? new Error(`Request failed: ${url}`);
}

async function fetchJson(url) {
  const response = await fetchWithRetry(url, { accept: GITHUB_API_ACCEPT });
  return response.json();
}

async function fetchText(url, accept = "text/html") {
  const response = await fetchWithRetry(url, { accept });
  return response.text();
}

async function fetchBuffer(url, accept = "*/*") {
  const response = await fetchWithRetry(url, { accept });
  const arrayBuffer = await response.arrayBuffer();
  return Buffer.from(arrayBuffer);
}

async function mapLimit(items, limit, worker) {
  const results = new Array(items.length);
  let index = 0;
  async function runner() {
    while (true) {
      const currentIndex = index;
      index += 1;
      if (currentIndex >= items.length) {
        return;
      }
      results[currentIndex] = await worker(items[currentIndex], currentIndex);
    }
  }
  const workers = Array.from({ length: Math.min(limit, items.length) }, () => runner());
  await Promise.all(workers);
  return results;
}

function extractFileBlocks(html) {
  const blocks = [];
  let searchFrom = 0;
  while (true) {
    const start = html.indexOf("<copilot-diff-entry", searchFrom);
    if (start === -1) {
      break;
    }
    const firstTagEnd = html.indexOf(">", start);
    if (firstTagEnd === -1) {
      break;
    }
    const closeTag = "</copilot-diff-entry>";
    const closeStart = html.indexOf(closeTag, firstTagEnd + 1);
    if (closeStart === -1) {
      break;
    }
    const end = closeStart + closeTag.length;
    const block = html.slice(start, end);
    if (block.includes('class="file js-file')) {
      blocks.push(block);
    }
    searchFrom = end;
  }
  return blocks;
}

function parseChangeCounts(block) {
  const match = block.match(
    /<span class="sr-only">\s*([\d,]+)\s+changes:\s*([\d,]+)\s+additions?\s*&amp;\s*([\d,]+)\s+deletions?\s*<\/span>/i,
  );
  if (!match) {
    return { changes: null, additions: null, deletions: null };
  }
  return {
    changes: Number.parseInt(match[1].replaceAll(",", ""), 10),
    additions: Number.parseInt(match[2].replaceAll(",", ""), 10),
    deletions: Number.parseInt(match[3].replaceAll(",", ""), 10),
  };
}

function parseDiffLines(block) {
  const lines = [];
  const addedFileLines = [];
  let sawAddition = false;
  let sawDeletion = false;
  let sawContext = false;
  const cellRegex = /<td class="([^"]*blob-code[^"]*)"[^>]*>([\s\S]*?)<\/td>/g;
  let match;
  while ((match = cellRegex.exec(block)) !== null) {
    const classes = match[1];
    const innerHtml = match[2];
    let kind = null;
    if (classes.includes("blob-code-hunk")) {
      kind = "hunk";
    } else if (classes.includes("blob-code-addition")) {
      kind = "addition";
      sawAddition = true;
    } else if (classes.includes("blob-code-deletion")) {
      kind = "deletion";
      sawDeletion = true;
    } else if (classes.includes("blob-code-context")) {
      kind = "context";
      sawContext = true;
    } else {
      continue;
    }

    let lineHtml = innerHtml;
    if (kind !== "hunk") {
      const spanMatch = innerHtml.match(
        /<span[^>]*class=['"][^'"]*blob-code-inner[^'"]*['"][^>]*>([\s\S]*?)<\/span>/i,
      );
      if (spanMatch) {
        lineHtml = spanMatch[1];
      }
    }

    const text = htmlInnerText(lineHtml);
    if (kind === "hunk") {
      lines.push(text);
      continue;
    }
    if (kind === "addition") {
      lines.push(`+${text}`);
      addedFileLines.push(text);
    } else if (kind === "deletion") {
      lines.push(`-${text}`);
    } else {
      lines.push(` ${text}`);
    }
  }
  return {
    diffText: lines.join("\n"),
    addedFileText: addedFileLines.join("\n"),
    sawAddition,
    sawDeletion,
    sawContext,
    firstHunk: lines.find((line) => line.startsWith("@@")) ?? null,
  };
}

function inferStatus({ deleted, additions, deletions, sawDeletion, sawContext, firstHunk }) {
  if (deleted) {
    return "deleted";
  }
  if (
    additions !== null &&
    deletions === 0 &&
    !sawDeletion &&
    !sawContext &&
    firstHunk &&
    firstHunk.startsWith("@@ -0,0 +")
  ) {
    return "added";
  }
  if (additions !== null && deletions === 0 && !sawDeletion && !sawContext) {
    return "added";
  }
  return "modified";
}

function parseFilesHtml(html) {
  const blocks = extractFileBlocks(html);
  const renderedFiles = blocks.map((block) => {
    const pathMatch =
      block.match(/\sdata-tagsearch-path="([^"]+)"/i) ??
      block.match(/class="file-header[\s\S]*?\sdata-path="([^"]+)"/i) ??
      block.match(/\sdata-path="([^"]+)"/i);
    const deletedMatch = block.match(/\sdata-file-deleted="([^"]+)"/i);
    const fileTypeMatch = block.match(/\sdata-file-type="([^"]*)"/i);
    const counts = parseChangeCounts(block);
    const diff = parseDiffLines(block);
    const filePath = pathMatch ? decodeHtmlEntities(pathMatch[1]) : null;
    const deleted = deletedMatch ? deletedMatch[1] === "true" : false;
    return {
      path: filePath,
      deleted,
      file_type: fileTypeMatch ? decodeHtmlEntities(fileTypeMatch[1]) : null,
      ...counts,
      diff_text: diff.diffText,
      added_file_text: diff.addedFileText,
      inferred_status: inferStatus({
        deleted,
        additions: counts.additions,
        deletions: counts.deletions,
        sawDeletion: diff.sawDeletion,
        sawContext: diff.sawContext,
        firstHunk: diff.firstHunk,
      }),
    };
  }).filter((file) => file.path);

  const renderedByPath = new Map(renderedFiles.map((file) => [file.path, file]));
  const fileTreePaths = [];
  const fileTreeRegex = /<span data-filterable-item-text hidden>([\s\S]*?)<\/span>/g;
  let treeMatch;
  while ((treeMatch = fileTreeRegex.exec(html)) !== null) {
    const treePath = decodeHtmlEntities(treeMatch[1].trim());
    if (treePath.includes("/")) {
      fileTreePaths.push(treePath);
    }
  }

  const mergedPaths = Array.from(new Set([...fileTreePaths, ...renderedFiles.map((file) => file.path)]));
  return mergedPaths.map((filePath) => {
    const rendered = renderedByPath.get(filePath);
    if (rendered) {
      return rendered;
    }
    return {
      path: filePath,
      deleted: false,
      file_type: path.extname(filePath) || null,
      changes: null,
      additions: null,
      deletions: null,
      diff_text: "",
      added_file_text: "",
      inferred_status: "unknown",
    };
  });
}

async function listPullRequests(options, cacheDir) {
  const prs = [];
  for (let page = 1; ; page += 1) {
    const url = `https://api.github.com/repos/${options.owner}/${options.repo}/pulls?state=${options.state}&per_page=100&page=${page}`;
    const data = await fetchJson(url);
    const cachePath = path.join(cacheDir, `pull_requests_page_${page}.json`);
    await writeText(cachePath, `${JSON.stringify(data, null, 2)}\n`);
    if (!Array.isArray(data) || data.length === 0) {
      break;
    }
    prs.push(...data);
    if (data.length < 100) {
      break;
    }
  }
  if (options.maxPrs != null) {
    return prs.slice(0, options.maxPrs);
  }
  return prs;
}

function slimPr(pr) {
  return {
    number: pr.number,
    state: pr.state,
    draft: Boolean(pr.draft),
    title: pr.title,
    author_login: pr.user?.login ?? null,
    html_url: pr.html_url,
    created_at: pr.created_at,
    updated_at: pr.updated_at,
    closed_at: pr.closed_at,
    merged_at: pr.merged_at,
    head_sha: pr.head?.sha ?? null,
    head_ref: pr.head?.ref ?? null,
    head_repo_full_name: pr.head?.repo?.full_name ?? null,
    head_repo_html_url: pr.head?.repo?.html_url ?? null,
    base_ref: pr.base?.ref ?? null,
    base_sha: pr.base?.sha ?? null,
  };
}

async function downloadFileContent(prMeta, file, directories) {
  if (file.deleted || !prMeta.head_repo_full_name || !prMeta.head_sha) {
    return {
      content_source: null,
      local_content_path: null,
      raw_bytes: null,
      sha256: null,
      fetch_error: null,
    };
  }

  const relativeFilePath = path.join(`pr_${prMeta.number}`, file.path);
  const outputPath = safeResolve(directories.filesDir, relativeFilePath);
  const rawUrl = `https://raw.githubusercontent.com/${prMeta.head_repo_full_name}/${prMeta.head_sha}/${encodeGitHubPath(file.path)}`;

  try {
    const buffer = await fetchBuffer(rawUrl);
    await writeBuffer(outputPath, buffer);
    return {
      content_source: "raw_head",
      local_content_path: path.relative(directories.outDirAbs, outputPath),
      raw_bytes: buffer.length,
      sha256: crypto.createHash("sha256").update(buffer).digest("hex"),
      fetch_error: null,
    };
  } catch (error) {
    if (file.inferred_status === "added" && file.added_file_text) {
      const reconstructed = `${file.added_file_text}\n`;
      const buffer = Buffer.from(reconstructed, "utf8");
      await writeBuffer(outputPath, buffer);
      return {
        content_source: "diff_reconstructed",
        local_content_path: path.relative(directories.outDirAbs, outputPath),
        raw_bytes: buffer.length,
        sha256: crypto.createHash("sha256").update(buffer).digest("hex"),
        fetch_error: String(error.message ?? error),
      };
    }
    return {
      content_source: null,
      local_content_path: null,
      raw_bytes: null,
      sha256: null,
      fetch_error: String(error.message ?? error),
    };
  }
}

async function processSinglePr(pr, directories) {
  const prMeta = slimPr(pr);
  const htmlUrl = `https://github.com/${directories.owner}/${directories.repo}/pull/${pr.number}/files`;
  const html = await fetchText(htmlUrl, "text/html");
  const htmlCachePath = path.join(directories.htmlCacheDir, `pr_${pr.number}.html`);
  await writeText(htmlCachePath, html);
  const files = parseFilesHtml(html);

  const fileRows = [];
  for (const file of files) {
    let localDiffPath = null;
    if (file.diff_text) {
      const diffRelativePath = path.join("diffs", `pr_${pr.number}`, `${file.path}.diff.txt`);
      const diffOutputPath = safeResolve(directories.outDirAbs, diffRelativePath);
      await writeText(diffOutputPath, `${file.diff_text}\n`);
      localDiffPath = path.relative(directories.outDirAbs, diffOutputPath);
    }

    const contentResult = await downloadFileContent(prMeta, file, directories);

    fileRows.push({
      pr_number: prMeta.number,
      pr_state: prMeta.state,
      pr_title: prMeta.title,
      pr_author_login: prMeta.author_login,
      head_sha: prMeta.head_sha,
      head_repo_full_name: prMeta.head_repo_full_name,
      path: file.path,
      inferred_status: file.inferred_status,
      deleted: file.deleted,
      additions: file.additions,
      deletions: file.deletions,
      changes: file.changes,
      file_type: file.file_type,
      local_diff_path: localDiffPath,
      local_content_path: contentResult.local_content_path,
      content_source: contentResult.content_source,
      raw_bytes: contentResult.raw_bytes,
      sha256: contentResult.sha256,
      raw_fetch_error: contentResult.fetch_error,
    });
  }

  return {
    pr: {
      ...prMeta,
      extracted_files: files.length,
    },
    files: fileRows,
  };
}

async function writeJsonl(filePath, rows) {
  const content = rows.map((row) => JSON.stringify(row)).join("\n");
  await writeText(filePath, content ? `${content}\n` : "");
}

async function writeCsv(filePath, rows, columns) {
  const lines = [columns.join(",")];
  for (const row of rows) {
    lines.push(columns.map((column) => csvEscape(row[column])).join(","));
  }
  await writeText(filePath, `${lines.join("\n")}\n`);
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const outDirAbs = path.resolve(options.outDir);
  const cacheDir = path.join(outDirAbs, "cache");
  const apiCacheDir = path.join(cacheDir, "api");
  const htmlCacheDir = path.join(cacheDir, "pull_files_html");
  const filesDir = path.join(outDirAbs, "files");

  await Promise.all([ensureDir(outDirAbs), ensureDir(apiCacheDir), ensureDir(htmlCacheDir), ensureDir(filesDir)]);

  console.log(`Fetching PR index for ${options.owner}/${options.repo}...`);
  const prIndex = await listPullRequests(options, apiCacheDir);
  console.log(`Found ${prIndex.length} PRs.`);

  const directories = {
    owner: options.owner,
    repo: options.repo,
    outDirAbs,
    htmlCacheDir,
    filesDir,
  };

  let processed = 0;
  const startedAt = new Date().toISOString();
  const processedResults = await mapLimit(prIndex, options.concurrency, async (pr) => {
    const result = await processSinglePr(pr, directories);
    processed += 1;
    if (processed % 10 === 0 || processed === prIndex.length) {
      console.log(`Processed ${processed}/${prIndex.length} PRs`);
    }
    return result;
  });

  const prRows = processedResults.map((entry) => entry.pr).sort((a, b) => a.number - b.number);
  const fileRows = processedResults
    .flatMap((entry) => entry.files)
    .sort((a, b) => (a.pr_number - b.pr_number) || a.path.localeCompare(b.path));

  const summary = {
    repository: `${options.owner}/${options.repo}`,
    state: options.state,
    started_at: startedAt,
    completed_at: new Date().toISOString(),
    pr_count: prRows.length,
    open_pr_count: prRows.filter((pr) => pr.state === "open").length,
    closed_pr_count: prRows.filter((pr) => pr.state === "closed").length,
    file_change_count: fileRows.length,
    added_file_count: fileRows.filter((file) => file.inferred_status === "added").length,
    deleted_file_count: fileRows.filter((file) => file.inferred_status === "deleted").length,
    content_download_count: fileRows.filter((file) => file.local_content_path).length,
    content_download_failure_count: fileRows.filter(
      (file) => !file.deleted && file.local_content_path == null,
    ).length,
  };

  await writeText(path.join(outDirAbs, "summary.json"), `${JSON.stringify(summary, null, 2)}\n`);
  await writeJsonl(path.join(outDirAbs, "prs.jsonl"), prRows);
  await writeJsonl(path.join(outDirAbs, "file_changes.jsonl"), fileRows);

  await writeCsv(path.join(outDirAbs, "prs.csv"), prRows, [
    "number",
    "state",
    "draft",
    "title",
    "author_login",
    "created_at",
    "updated_at",
    "closed_at",
    "merged_at",
    "head_sha",
    "head_ref",
    "head_repo_full_name",
    "base_ref",
    "base_sha",
    "html_url",
    "extracted_files",
  ]);

  await writeCsv(path.join(outDirAbs, "file_changes.csv"), fileRows, [
    "pr_number",
    "pr_state",
    "pr_title",
    "pr_author_login",
    "head_sha",
    "head_repo_full_name",
    "path",
    "inferred_status",
    "deleted",
    "additions",
    "deletions",
    "changes",
    "file_type",
    "content_source",
    "raw_bytes",
    "sha256",
    "local_content_path",
    "local_diff_path",
    "raw_fetch_error",
  ]);

  console.log("Done.");
  console.log(JSON.stringify(summary, null, 2));
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exit(1);
});
