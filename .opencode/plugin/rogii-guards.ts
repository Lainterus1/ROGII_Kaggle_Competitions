import { execFileSync } from "node:child_process"

const RUNTIME_OR_SECRET_PATTERNS = [
  /^data[\\/]/i,
  /^outputs[\\/]/i,
  /^models[\\/]/i,
  /^submissions[\\/]/i,
  /^mlruns[\\/]/i,
  /^kaggle_datasets[\\/]/i,
  /^wheels[\\/]/i,
  /(^|[\\/])kaggle\.json$/i,
  /(^|[\\/])\.env$/i,
  /\.(csv|parquet|feather|pkl|joblib|model|cbm|lgb|xgb)$/i,
]

const LEAKAGE_SENSITIVE_PATHS = [
  /^src[\\/]rogii[\\/]features\.py$/,
  /^src[\\/]rogii[\\/]train\.py$/,
  /^src[\\/]rogii[\\/]validation\.py$/,
  /^src[\\/]rogii[\\/]predict\.py$/,
  /^scripts[\\/]run_train\.py$/,
  /^scripts[\\/]run_predict\.py$/,
  /^tests[\\/]test_no_target_leakage\.py$/,
  /^tests[\\/]test_validation_split\.py$/,
]

const CODE_PATH = /^(src|scripts|tests)[\\/].*\.py$/

function normalizePath(path: string) {
  return path.trim().replace(/^"|"$/g, "").replace(/^'|'$/g, "").replace(/\\/g, "/")
}

function isRuntimeOrSecret(path: string) {
  const normalized = normalizePath(path)
  return RUNTIME_OR_SECRET_PATTERNS.some((pattern) => pattern.test(normalized))
}

function isLeakageSensitive(path: string) {
  const normalized = normalizePath(path)
  return LEAKAGE_SENSITIVE_PATHS.some((pattern) => pattern.test(normalized))
}

function isCodePath(path: string) {
  return CODE_PATH.test(normalizePath(path))
}

function runGit(args: string[]) {
  try {
    return execFileSync("git", args, { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] })
  } catch (_error) {
    return ""
  }
}

function splitNullDelimited(output: string) {
  return output.split("\0").map((item) => item.trim()).filter(Boolean)
}

function stagedFiles() {
  return splitNullDelimited(runGit(["diff", "--cached", "--name-only", "-z"]))
}

function trackedFiles() {
  return splitNullDelimited(runGit(["ls-files", "-z"]))
}

function assertNoStagedRuntimeArtifacts() {
  const risky = stagedFiles().filter(isRuntimeOrSecret)
  if (risky.length === 0) return
  throw new Error(
    [
      "ROGII guard blocked git commit: staged runtime artifacts, data, models, submissions or secrets were detected.",
      "Unstage or remove these paths before committing:",
      ...risky.map((path) => `- ${path}`),
    ].join("\n"),
  )
}

function assertNoTrackedRuntimeArtifacts() {
  const risky = trackedFiles().filter(isRuntimeOrSecret)
  if (risky.length === 0) return
  throw new Error(
    [
      "ROGII guard blocked git push: runtime artifacts, data, models, submissions or secrets are tracked by Git.",
      "Remove them from the index/history policy before pushing:",
      ...risky.map((path) => `- ${path}`),
    ].join("\n"),
  )
}

function assertKaggleSubmitApproved(command: string) {
  const lower = command.toLowerCase()
  if (!/\bkaggle\s+competitions\s+submit\b/.test(lower)) return

  const approved =
    /ROGII_ALLOW_KAGGLE_SUBMIT\s*=\s*1/.test(command) ||
    /\$env:ROGII_ALLOW_KAGGLE_SUBMIT\s*=\s*["']?1["']?/i.test(command)

  if (approved) return

  throw new Error(
    [
      "ROGII guard blocked Kaggle submission.",
      "Kaggle submissions are manual and require explicit user approval plus prior submission validation.",
      "After approval, rerun with ROGII_ALLOW_KAGGLE_SUBMIT=1 in the command environment.",
    ].join("\n"),
  )
}

function patchPaths(patchText: string) {
  const paths = []
  for (const line of patchText.split(/\r?\n/)) {
    const match = line.match(/^\*\*\* (?:Add|Update|Delete) File: (.+)$/) || line.match(/^\*\*\* Move to: (.+)$/)
    if (match) paths.push(normalizePath(match[1]))
  }
  return paths
}

function reminderPrinter() {
  const shown = new Set<string>()
  return (key: string, lines: string[]) => {
    if (shown.has(key)) return
    shown.add(key)
    console.warn(lines.map((line) => `[rogii-guard] ${line}`).join("\n"))
  }
}

export default async function RogiiGuards() {
  const remind = reminderPrinter()

  return {
    "tool.execute.before": async (input: any, output: any) => {
      const tool = input && input.tool
      const args = (output && output.args) || (input && input.args) || {}

      if (tool === "bash") {
        const command = String(args.command || "")
        const lower = command.toLowerCase()

        assertKaggleSubmitApproved(command)

        if (/\bgit\s+commit\b/.test(lower)) assertNoStagedRuntimeArtifacts()
        if (/\bgit\s+push\b/.test(lower)) assertNoTrackedRuntimeArtifacts()

        if (/\brun_predict\.py\b/.test(lower) || /submission\.csv\b/.test(lower)) {
          remind("submission-validation", [
            "Submission-related command detected.",
            "Validate generated output with: python scripts/validate_submission.py --data-dir data --submission <path>",
            "Do not submit to Kaggle without explicit user approval.",
          ])
        }

        if (/\brun_train\.py\b/.test(lower) || /--include-|--residual-target|--eval-postproc/.test(lower)) {
          remind("experiment-logging", [
            "Training or experiment command detected.",
            "If this produces a real CV/LB result, update docs/EXPERIMENT_LOG.md and relevant roadmap/task docs.",
          ])
        }
      }

      if (tool === "apply_patch") {
        const paths = patchPaths(String(args.patchText || ""))
        if (paths.some(isRuntimeOrSecret)) {
          remind("runtime-path-edit", [
            "Patch touches runtime/artifact paths. Keep raw data, models, submissions, mlruns, wheels and secrets out of Git.",
          ])
        }

        if (paths.some(isLeakageSensitive)) {
          remind("leakage-sensitive-edit", [
            "Leakage-sensitive files changed.",
            "Run or justify: python -m pytest tests/test_no_target_leakage.py tests/test_validation_split.py",
            "Review TVT_input, target-like columns, GroupKFold overlap and OOF reference construction before promotion/submission.",
          ])
        } else if (paths.some(isCodePath)) {
          remind("code-edit", ["Python code changed. Run the relevant tests; default project check is: python -m pytest tests"])
        }
      }
    },
  }
}
