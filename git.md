- [1. Pull Requests and Issues](#1-pull-requests-and-issues)
  - [1.1. Pull Requests](#11-pull-requests)
  - [1.2. Issues](#12-issues)
- [2. Forking](#2-forking)

## 1. Pull Requests and Issues

To get started with the GitHub in VS Code, you'll need to install Git, create a GitHub account and install the GitHub Pull Requests and Issues extension.

You can search for and clone a repository from GitHub using the Git: Clone command in the Command Palette (Ctrl+Shift+P) or by using the Clone Repository button in the Source Control view (available when you have no folder open).

Note: If you'd like to work on a repository without cloning the contents to your local machine, you can install the GitHub Repositories extension to browse and edit directly on GitHub. You can learn more below in the GitHub Repositories extension section.

User suggestions are triggered by the "@" character and issue suggestions are triggered by the "#" character. Suggestions are available in the editor and in the Source Control view's input box.

### 1.1. Pull Requests

Visual Studio Code also supports pull request workflows through the GitHub Pull Requests and Issues extension available on the VS Code Marketplace. Pull request extensions let you review, comment, and verify source code contributions directly within VS Code.

**Create Pull Requests**

Once you have committed changes to your fork or branch, you can use the GitHub Pull Requests: Create Pull Request command or the Create Pull Request button in the Pull Requests view to create a pull request.

A new Create Pull Request view will be displayed where you can select the repository and branch you'd like your pull request to target as well as fill in details such as the title, description, and whether it is a draft PR. If your repository has a pull request template, this will automatically be used for the description.

![Create Pull Request](https://code.visualstudio.com/assets/docs/editor/github/create-pull-request-view.png)

Once you select Create, if you have not already pushed your branch to a GitHub remote, the extension will ask if you'd like to publish the branch and provides a dropdown to select the specific remote.

The Create Pull Request view now enters Review Mode, where you can review the details of the PR, add comments, reviewers, and labels, and merge the PR once it's ready.

After the PR is merged, you'll have the option to delete both the remote and local branch.

**Reviewing**

Pull requests can be reviewed from the Pull Requests view. You can assign reviewers and labels, add comments, approve, close, and merge all from the pull request Description.

![Pull Request Description Editor](https://code.visualstudio.com/assets/docs/editor/github/pull-request-description-editor.png)

Pull Request Description editor
From the Description page, you can also easily checkout the pull request locally using the Checkout button. This will switch VS Code to open the fork and branch of the pull request (visible in the Status bar) in Review Mode and add a new Changes in Pull Request view from which you can view diffs of the current changes as well as all commits and the changes within these commits. Files that have been commented on are decorated with a diamond icon. To view the file on disk, you can use the Open File inline action.

![Changes in Pull Request view](https://code.visualstudio.com/assets/docs/editor/github/changes-view.png)

The diff editors from this view use the local file, so file navigation, IntelliSense, and editing work as normal. You can add comments within the editor on these diffs. Both adding single comments and creating a whole review is supported.

When you are done reviewing the pull request changes you can merge the PR or select Exit Review Mode to go back to the previous branch you were working on.

### 1.2. Issues

**Creating Issues**

Issues can be created from the + button in the Issues view and by using the GitHub Issues: Create Issue from Selection and GitHub Issues: Create Issue from Clipboard commands. They can also be created using a Code Action for "TODO" comments. When creating issues, you can take the default description or select the Edit Description pencil icon in the upper right to bring up an editor for the issue body.

![Create Issue](https://code.visualstudio.com/assets/docs/editor/github/issue-from-todo.gif)

You can configure the trigger for the Code Action using the GitHub Issues: Create Issue Triggers (githubIssues.createIssueTriggers) setting.

The default issue triggers are:

```markdown
"githubIssues.createIssueTriggers": [
  "TODO",
  "todo",
  "BUG",
  "FIXME",
  "ISSUE",
  "HACK"
]
```

**Working on issues**

From the Issues view, you can see your issues and work on them.

![Issues view](https://code.visualstudio.com/assets/docs/editor/github/issues-view.png)

By default, when you start working on an issue (Start Working on Issue context menu item), a branch will be created for you, as shown in the Status bar.

The Status bar also shows the active issue and if you select that item, a list of issue actions are available such as opening the issue on the GitHub website or creating a pull request.

Once you are done working on the issue and want to commit a change, the commit message input box in the Source Control view will be populated.

## 2. Forking

Branching is to create another line of development in the project without affecting the main branch or repository. Forking, on the other hand, is to make a clone of the repository on your GitHub account without affecting the main repository. 
