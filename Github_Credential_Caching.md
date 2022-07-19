# Github Credential Caching

Since August 2021 Github is requiring its users to use a PAT to log into remote and local machines [1]. This makes it so, at least on accounts with two-factor authentification (2FA), always ask for your credentials when sync processes are attempted (pushing, fetching, etc). This behavior is met using the git process from console or VScode extensions such as `GitHub Pull Requests and Issues`.

This is a short guide on how to cache credentials and PATs on your system. 

## Possible solutions:

### 1. Github Desktop (Windows)

The Github Desktop client can save credentials and allow normal usage. It has no official linux support. 

### 2. Github Credential Manager Core (Windows)

Github Credential Manager Core can be used to sacurely store credentials. It has no official linux support.

### 3. Github CLI (Linux)

Install github CLI [2] that is officially supported to securely keep credentials.

```
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null
sudo apt update
sudo apt install gh
```

Then use the command to log in: 

```
gh auth login
```

All set.


Links:
- [1] - https://github.blog/2020-12-15-token-authentication-requirements-for-git-operations/
- [2] - https://github.com/cli/cli#installation
