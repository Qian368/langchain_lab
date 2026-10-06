/* 初始化创建 MongoDB 的程序业务账号 */
/* global use */
// MongoDB Playground
// Use Ctrl+Space inside a snippet or a string literal to trigger completions.

// The current database to use.
use("knowledge_base");

// 创建程序侧账号 app_user
db.createUser({
    user: "app_user",
    pwd: "AppPass123456",
    roles: [
        // ✅ 仅对 knowledge_base 库读写，不能删库、不能管理用户、不能修改库配置
        { role: "readWrite", db: "knowledge_base" }
    ]
});


