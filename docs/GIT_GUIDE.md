# 每天够用的 Git

Git 保存本地修改历史；GitHub 保存远程副本。`commit` 不等于上传，`push` 才会把提交发送到 GitHub。

## 第一次上传

使用仓库外附带的 `Publish-ToGitHub.ps1`。它只复制缺失文件，遇到内容不同的同名文件会停止。默认目标为 `D:\code\26fall_AiLearning`，GitHub 仓库名为 `26fall_AiLearning`，可见性为公开。

如果需要登录，按终端提示在浏览器中完成 GitHub 授权。不要把密码、访问令牌或一次性登录码写进代码或周报。

## 平时保存与上传

在 PowerShell 中进入仓库：

```powershell
cd D:\code\26fall_AiLearning
git status
git diff
git add weekly/week01.md
git diff --cached
git commit -m "docs: update week 1 progress"
git push
```

将文件路径换成实际修改的文件。先查看差异再提交；不要为了提交次数拆碎工作。

- `status`：哪些文件改变了。
- `diff`：尚未暂存的具体修改。
- `add`：选择本次要保存的文件。
- `diff --cached`：检查本次提交的内容。
- `commit`：保存一份本地历史。
- `push`：上传本地提交。
- `log --oneline -5`：查看最近五次提交。

## 两台电脑如何同步

第二台电脑用 `git clone 仓库地址` 获取仓库，不重新初始化。

每次开始学习前，在没有未提交修改时运行 `git pull --ff-only`。结束后提交并上传，再切换到另一台电脑。

若 pull 提示无法快进或出现冲突，先停止并查看 status。保留现场求助，不运行强制覆盖命令。

## 第 2 周练一次分支

```powershell
git switch -c practice/add-search
# 修改并检查文件后，add 和 commit
git switch main
git merge practice/add-search
git push
```

## 公开上传前检查

- 只放自己的学习材料，引用代码注明出处。
- API 密钥放 `.env`，不要提交真实值。
- 模型、数据、虚拟环境不上传；小图表可选择性放在 `reports/`。
- `.gitignore` 不会自动移除已经提交过的文件。
- 不提交私人信息、未获准公开的实验室材料和账号凭据。
