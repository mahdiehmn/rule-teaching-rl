"""
Load the development job's credentials as data, then execute its command.

Dotenv files can have CRLF line endings and dotenv quoting; they are not
shell programs. Never source them or print their contents in Slurm logs.
The same loader is used for preflight and the actual frozen-code run.
"""

import argparse
import os
from pathlib import Path
import subprocess

from dotenv import dotenv_values


def environment(env_file, inherited=None):
    """
    Preserve an exported key, or read only that key from the dotenv file.
    """

    result = dict(os.environ if inherited is None else inherited)
    key = result.get('OPENAI_API_KEY', '').strip()
    if not key and Path(env_file).is_file():
        values = dotenv_values(env_file, encoding='utf-8-sig',
                               interpolate=False)
        key = (values.get('OPENAI_API_KEY') or '').strip()
    if not key:
        raise ValueError('OPENAI_API_KEY is missing from the environment '
                         'and the repository .env file')
    result['OPENAI_API_KEY'] = key
    # Pin the provider after reading credentials. A dotenv file from an
    # Aleph experiment must not redirect this OpenAI development run.
    result['LLM_PROVIDER'] = 'openai'
    result['MODEL'] = 'gpt-5-mini'
    # This must be set BEFORE the trainer constructs its SDK clients.
    # Setting it only when reserving later cannot change existing clients.
    result['LLM_MAX_SDK_RETRIES'] = '0'
    return result


def main():
    """
    Check credentials without requests, or pass them to a child process.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', required=True)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    try:
        env = environment(args.env_file)
    except ValueError as exc:
        parser.exit(1, f'{exc}\n')
    if args.check:
        print('Credentials: present; provider: openai; SDK retries: 0')
        return
    command = args.command
    if command[:1] == ['--']:
        command = command[1:]
    if not command:
        parser.error('supply --check or a command after --')
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == '__main__':
    main()
