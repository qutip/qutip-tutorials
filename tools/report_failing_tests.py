
import requests
import json
import sys
import argparse
from datetime import date

def open_issue(token):
    url = "https://api.github.com/repos/qutip/qutip-jax/issues"
    data = json.dumps({
        "title": f"Automated tests failed on {date.today()}",
        "labels": ["bug"],
        "body": "Scheduled test failed!"
    })

    headers = {
        "Accept": "application/vnd.github.v3+json",
        "Authorization" : f"token {token}",
    }

    post_request = requests.post(url=url, data=data, headers=headers)

    if post_request.status_code == 201:
        print("Success")

    else:
        print(
            "Fail:",
            post_request.status_code,
            post_request.reason,
            post_request.content
        )


def main():
    parser = argparse.ArgumentParser(
        description="""Open an issue on failed tests."""
    )
    parser.add_argument("token")
    args = parser.parse_args()
    print(args.token)
    open_issue(args.token)


if __name__ == "__main__":
    sys.exit(main())
