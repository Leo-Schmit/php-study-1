#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import shutil
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print("[main] ERROR: GITHUB_TOKEN environment variable is not set.")
    sys.exit(1)

headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v4+json"
}

search_query = "language:PHP stars:>1500 archived:false sort:stars-desc"
per_page = 100
work_dir = Path("/app/workspace")
output_file = Path("combined_repo_data.json")

session = requests.Session()
session.headers.update(headers)
print(f"[main] Session headers: {session.headers}")

retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[500, 502, 503, 504]
)
session.mount("https://", HTTPAdapter(max_retries=retry_strategy))


def check_exists_analyzer(repo_full_name):
    """
    Check if the repository contains any supported analyzer config (PHPStan, Psalm, or Phan).
    """
    url = f"https://api.github.com/repos/{repo_full_name}/contents"
    print(f"[check_exists_analyzer] Checking analyzer for {repo_full_name}")
    try:
        response = session.get(url)
        if response.status_code != 200:
            print(
                f"[check_exists_analyzer] Failed to fetch contents: {response.status_code}")
            return False
        items = response.json()
        names = [item["name"].lower()
                 for item in items if isinstance(item, dict) and "name" in item]

        if any(name.startswith("phpstan.") for name in names):
            return True
        if any(name.startswith("psalm") for name in names):
            return True
        if ".phan" in names:
            return True

        return False

    except Exception as e:
        print(f"[check_exists_analyzer] Error: {e}")
        return False


def fetch_repositories():
    """
    Retrieve up to 1000 PHP repositories with more than 1500 stars using REST API.
    """
    def paginate(query, max_pages=10):
        repos = []
        page = 1

        while page <= max_pages:
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page
            }
            url = "https://api.github.com/search/repositories"
            print(f"[fetch_repositories] Fetching page {page}")
            try:
                resp = session.get(url, params=params)
                resp.raise_for_status()
                items = resp.json().get("items", [])
                repos.extend(item["full_name"] for item in items)
                if len(items) < per_page:
                    break
                page += 1
            except Exception as e:
                print(f"[fetch_repositories] Error: {e}")
                break

        return repos

    repos = paginate(search_query)
    print(f"[fetch_repositories] Retrieved {len(repos)} repositories")
    return repos


def count_bug_issues(repo_full_name):
    owner, name = repo_full_name.split("/", 1)
    query = """
    query($owner: String!, $name: String!) {
      repository(owner: $owner, name: $name) {
        issues(labels: ["bug"]) {
          totalCount
        }
      }
    }
    """
    try:
        resp = session.post(
            "https://api.github.com/graphql",
            json={"query": query, "variables": {"owner": owner, "name": name}},
            headers={"Authorization": f"bearer {GITHUB_TOKEN}"}
        )
        resp.raise_for_status()
        repo_data = resp.json().get("data", {}).get("repository", {})
        total_bugs = repo_data.get("issues", {}).get("totalCount", 0)
        return {"total_bugs": total_bugs}
    except Exception as e:
        print(f"[count_bug_issues] Error for {repo_full_name}: {e}")
        return {"total_bugs": 0}


def run_phpstan(local_dir):
    print(f"[run_phpstan] Running PHPStan for {local_dir}")
    (local_dir / "phpstan.neon").write_text(
        "parameters:\n  paths:\n    - .\n  excludePaths:\n    - **/Tests/*\n    - **/tests/*\n    - **/e2e/*\n    - vendor/*\n", encoding="utf-8"
    )

    cmd = [
        "phpstan", "analyze", ".",
        "--debug", "--level=max",
        "--no-progress", "--error-format=table"
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=local_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
    except FileNotFoundError:
        print(f"[run_phpstan] Command 'phpstan' not found. Please install it and make sure it's in PATH.")
        return
    except Exception as e:
        print(f"[run_phpstan] Failed to start PHPStan: {e}")
        return
    output = []
    try:
        while True:
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if line:
                print(line, end="")
                output.append(line)
        proc.wait()
    except Exception as e:
        print(f"[run_phpstan] Error while reading PHPStan output: {e}")
        return

    errors = {}
    for line in output:
        if '🪪' in line:
            parts = line.strip().split()
            identifier = parts[-1]
            errors[identifier] = errors.get(identifier, 0) + 1

    if errors:
        return dict(sorted(errors.items(), key=lambda x: x[1], reverse=True))
    else:
        print(f"[run_phpstan] No errors found by PHPStan.")
        return


def run_psalm(local_dir):
    print(f"[run_psalm] Running Psalm for {local_dir}")
    cfg = Path(local_dir) / 'psalm.xml'
    report_path = Path(local_dir) / 'psalm.json'

    cfg.write_text(
        """<?xml version="1.0"?>
<psalm errorLevel="1">
    <projectFiles>
        <directory name="." />
        <ignoreFiles allowMissingFiles="true">
            <directory name="vendor/" />
            <directory name="tests/" />
            <directory name="e2e/" />
            <directory name="Tests/" />
        </ignoreFiles>
    </projectFiles>
</psalm>
""",
        encoding="utf-8"
    )

    cmd = [
        "psalm",
        "--output-format=json",
        "--show-info=false",
        f"--report={report_path}",
    ]

    try:
        subprocess.run(
            cmd,
            cwd=local_dir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"[run_psalm] Failed to run Psalm for {local_dir}: {e}")
        return
    except FileNotFoundError:
        print(f"[run_psalm] Command 'psalm' not found. Please make sure it is installed and in PATH.")
        return

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        result = {}
        for issue in report:
            issue_type = issue.get("type", "unknown")
            result[issue_type] = result.get(issue_type, 0) + 1

        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
    except FileNotFoundError:
        print(f"[run_psalm] Report file '{report_path}' not found after running Psalm.")
        return
    except json.JSONDecodeError as e:
        print(f"[run_psalm] Failed to parse JSON report: {e}")
        return


def run_phan(local_dir):
    print(f"[run_phan] Running Phan for {local_dir}")
    cfg_dir = Path(local_dir) / '.phan'
    cfg_dir.mkdir(parents=True, exist_ok=True)

    cfg = cfg_dir / 'config.php'
    cfg.write_text(
        """<?php
    use Phan\Issue;

    return [
        'minimum_severity'=> Issue::SEVERITY_LOW,
        'directory_list' => [
            '.',
        ],
        'exclude_analysis_directory_list' => [
            'vendor/',
            'tests/',
            'e2e/',
        ],
    ];
    """,
        encoding="utf-8"
    )

    cmd = ["phan", "--output-mode=json",
           "--no-progress-bar", "--no-color", "--analyze-twice"]
    proc = subprocess.Popen(
        cmd, cwd=local_dir,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace"
    )
    out, _ = proc.communicate()
    if proc.returncode != 0 and not out.strip():
        print(f"[run_phan] Phan failed for {local_dir} with empty output")
        return
    try:
        issues = json.loads(out)
        counts = {}
        for issue in issues:
            issue_type = issue.get("check_name", "unknown")
            counts[issue_type] = counts.get(issue_type, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
    except FileNotFoundError:
        print(
            f"[run_phan] Command 'phan' not found. Please install it and make sure it's in PATH.")
        return
    except Exception as e:
        print(f"[run_phan] Failed to run Phan: {e}")
        return


def analyze_repository(repo_full_name, existing=None):
    print(f"[analyze_repository] Analyzing {repo_full_name}")
    local_dir = work_dir / repo_full_name.replace("/", "__")
    if local_dir.exists():
        shutil.rmtree(local_dir)
    clone = subprocess.run(
        ["git", "clone", "--depth", "1",
            f"https://github.com/{repo_full_name}.git", str(local_dir)],
        capture_output=True, text=True
    )
    if clone.returncode != 0 or not local_dir.exists():
        return {}

    cloc = subprocess.run(
        ["cloc", str(local_dir), "--include-lang=PHP",
         "--json", "--timeout=0"],
        capture_output=True, text=True
    )
    loc = 0
    if cloc.returncode == 0:
        try:
            loc = json.loads(cloc.stdout).get("PHP", {}).get("code", 0)
        except json.JSONDecodeError:
            pass

    try:
        subprocess.run(
            ["composer", "install", "--ignore-platform-reqs", "--no-scripts", "--no-interaction"],
            check=True, cwd=local_dir
        )
    except subprocess.CalledProcessError as e:
        print(f"[analyze_repository] Composer install error: {e}")

    results = {"lines_of_code": loc}
    existing = existing or {}

    # Phan
    phan_output = run_phan(local_dir)
    if phan_output is not None:
        results["phan_results"] = phan_output

    if "psalm_results" not in existing:
        psalm_output = run_psalm(local_dir)
        if psalm_output is not None:
            results["psalm_results"] = psalm_output

    if "phpstan_results" not in existing:
        phpstan_output = run_phpstan(local_dir)
        if phpstan_output is not None:
            results["phpstan_results"] = phpstan_output

    shutil.rmtree(local_dir, ignore_errors=True)
    return results


def main():
    work_dir.mkdir(exist_ok=True)
    data = {}
    if output_file.exists():
        try:
            data = json.loads(output_file.read_text(encoding="utf-8")) or {}
        except Exception:
            data = {}

    repos = fetch_repositories()
    print(f"[main] Processing {len(repos)} repositories")
    for repo in repos:
        if repo == 'glpi-project/glpi' or repo == 'leafo/lessphp' or repo == 'vinkla/wordplate' or repo == 'roots/bedrock':
            continue
        if repo in data:
            resp = session.get(f"https://api.github.com/repos/{repo}")
            if resp.status_code == 200:
                stars = resp.json().get("stargazers_count")
                if stars is not None:
                    data[repo]["stars"] = stars
                    output_file.write_text(json.dumps(
                        data, indent=2), encoding="utf-8")

            has_phpstan = "phpstan_results" in data[repo]
            has_psalm = "psalm_results" in data[repo]
            has_phan = "phan_results" in data[repo]
            if has_phpstan and has_psalm and has_phan:
                continue

        if session.get(f"https://api.github.com/repos/{repo}/contents/composer.json").status_code != 200:
            continue

        github_bugs = count_bug_issues(repo)
        if github_bugs == 0:
            continue

        existing = data.get(repo, {})
        if existing.get('psalm_results') and existing.get('phpstan_results'):
            continue

        new_results = analyze_repository(repo, existing)
        if not new_results:
            continue

        entry = {
            **existing,
            **new_results,
            "github_bugs": github_bugs,
            "exists_analyzer": check_exists_analyzer(repo)
        }
        data[repo] = entry
        output_file.write_text(json.dumps(data, indent=2), encoding="utf-8")

        print(f"[main] Updated {repo}")

    print(f"Done! Output in {output_file}")


if __name__ == "__main__":
    main()
