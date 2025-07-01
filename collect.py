#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
import base64

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    print("[main] ERROR: GITHUB_TOKEN environment variable is not set.")
    sys.exit(1)

graphql_url = "https://api.github.com/graphql"
headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v4+json"
}

search_query = "language:PHP stars:>1500 archived:false sort:stars-desc"
per_page = 100
work_dir = Path("/app/workspace")
output_file = Path("/app/results/data.json")

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
    print(f"[check_exists_analyzer] Checking analyzer for {repo_full_name}")
    found = []
    try:
        contents_url = f"https://api.github.com/repos/{repo_full_name}/contents"
        response = session.get(contents_url)
        if response.status_code != 200:
            print(
                f"[check_exists_analyzer] Failed to fetch contents: {response.status_code}")
            return found
        items = response.json()
        names = [item.get("name", "").lower()
                 for item in items if isinstance(item, dict)]

        if any(name.startswith("phpstan.") for name in names):
            found.append("phpstan")
        if any(name.startswith("psalm") for name in names):
            found.append("psalm")
        if ".phan" in names:
            found.append("phan")

        comp_url = f"https://api.github.com/repos/{repo_full_name}/contents/composer.json"
        comp_resp = session.get(comp_url)
        if comp_resp.status_code == 200:
            comp_info = comp_resp.json()
            encoded = comp_info.get("content", "")
            decoded = base64.b64decode(encoded).decode('utf-8')
            composer = json.loads(decoded)
            dev_reqs = composer.get("require-dev", {}) or {}
            pkg_map = {
                "phpstan/phpstan": "phpstan",
                "vimeo/psalm": "psalm",
                "phan/phan": "phan"
            }
            for pkg, name in pkg_map.items():
                if pkg in dev_reqs and name not in found:
                    found.append(name)

        return found

    except Exception as e:
        print(f"[check_exists_analyzer] Error: {e}")
        return found


def fetch_repositories():
    repos = []
    cursor = None
    fetched = 0
    max_repos = 1000

    graphql_query = '''
    query($query: String!, $first: Int!, $after: String) {
      search(query: $query, type: REPOSITORY, first: $first, after: $after) {
        pageInfo {
          endCursor
          hasNextPage
        }
        nodes {
          ... on Repository {
            nameWithOwner
          }
        }
      }
    }
    '''

    while fetched < max_repos:
        batch_size = min(100, max_repos - fetched)
        variables = {"query": search_query,
                     "first": batch_size, "after": cursor}
        print(
            f"[fetch_repositories] Fetching {batch_size} repos after cursor {cursor}")
        response = session.post(
            graphql_url,
            json={"query": graphql_query, "variables": variables}
        )
        response.raise_for_status()
        data = response.json().get("data", {}).get("search", {})

        nodes = data.get("nodes", [])
        for node in nodes:
            repos.append(node["nameWithOwner"])

        fetched += len(nodes)
        page_info = data.get("pageInfo", {})
        if not page_info.get("hasNextPage") or not nodes:
            break
        cursor = page_info.get("endCursor")

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
        return total_bugs
    except Exception as e:
        print(f"[count_bug_issues] Error for {repo_full_name}: {e}")
        return 0


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
        print(
            f"[run_phpstan] Command 'phpstan' not found. Please install it and make sure it's in PATH.")
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
        print(
            f"[run_psalm] Command 'psalm' not found. Please make sure it is installed and in PATH.")
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
        print(
            f"[run_psalm] Report file '{report_path}' not found after running Psalm.")
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
            ["composer", "install", "--ignore-platform-reqs",
                "--no-scripts", "--no-interaction"],
            check=True, cwd=local_dir
        )
    except subprocess.CalledProcessError as e:
        print(f"[analyze_repository] Composer install error: {e}")

    results = {"lines_of_code": loc}
    existing = existing or {}

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
    repos = fetch_repositories()
    new_data = {}

    for repo in repos:
        resp = session.get(f"https://api.github.com/repos/{repo}")
        if resp.status_code != 200:
            continue

        stars = resp.json().get("stargazers_count")
        entry = {}
        if stars is not None:
            entry["stars"] = stars

        if session.get(f"https://api.github.com/repos/{repo}/contents/composer.json").status_code != 200:
            continue

        github_bugs = count_bug_issues(repo)
        if github_bugs == 0:
            continue

        new_results = analyze_repository(repo, entry)
        if not new_results:
            continue

        entry.update(new_results)
        entry["github_bugs"] = github_bugs
        entry["exists_analyzer"] = check_exists_analyzer(repo)

        new_data[repo] = entry
        output_file.write_text(json.dumps(
            new_data, indent=2), encoding="utf-8")
        print(f"Updated and persisted entry for {repo}")

    print(f"Done! Output written to {output_file}")


if __name__ == "__main__":
    main()
