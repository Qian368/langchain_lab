"""控制面 HTTP API：统一命令入口 + 指令发现

- POST /commands：{session_id, action, payload}，payload 按指令注册表 schema 校验
- GET /instructions：注册表公开契约（前端帮助 / 载荷表单）
- 身份锚点：session_id 唯一标识运行时会话，user/tenant 上下文在会话创建时由运行时绑定注入，
  请求不重复传身份字段（避免客户端自报身份与实例不一致）
"""
