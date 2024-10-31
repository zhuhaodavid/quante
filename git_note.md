这个 git 的结构是这样的

稳定的版本为 master 分支，开发版本为 dev 分支

一般情况下使用 dev 分支

如何需要同步库的内容按如下操作：

1. git checkout dev

2. 将 share 文件夹中的内容复制到 share 分支

3. add commit share

4. 
    git checkout share
    git merge dev

    手动解决冲突

5. 
    git checkout dev
    git merge share

这样就完成了一次更新
