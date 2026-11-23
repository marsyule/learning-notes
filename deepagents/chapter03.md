# Task03｜第三章：虚拟文件系统与存储后端

> **状态：学习中｜已整理课程知识框架，等待第一轮苏格拉底问答。**  
> 课程：DeepAgents In Action · Task03  
> 资料依据：[官方第三章：虚拟文件系统 — Deep Agents 的 Context Engineering 核心](https://github.com/datawhalechina/deepagents-in-action/blob/main/content/ch03-virtual-filesystem.md)  
> 说明：下文“课程知识”来自官方教程；“我的理解”尚未填写，后续依据问答更新。示例代码尚未由本人运行。

## 0. 这一章究竟要解决什么？

**核心问题：为什么 Agent 需要虚拟文件系统，而不是把所有信息都放进 Prompt？**

传统做法把搜索结果、文档和中间产物持续追加到对话历史，导致输入膨胀、注意力分散、重复消耗 Token，长程任务难以管理。Deep Agents 给 Agent 提供文件操作能力，使其能够**将信息保存为有路径的资源，只在需要时读取相关片段**。

**一句话心智模型：** 对话上下文是当前工作台；虚拟文件系统是可按需访问的资料柜。重要的是**控制哪些信息在什么时候进入模型上下文**，而不仅是“能写文件”。

### 理解框架

~~~text
用户任务 / Agent 推理
        │
        ├── 内置文件工具：ls / read_file / write_file / edit_file / delete / glob / grep
        │                            │
        │                      BackendProtocol
        │                            │
        │            State / Filesystem / Store / Composite / Sandbox ...
        │                            │
        └── 按需读取文件或保存中间结果 ─────→ 进入后续模型输入

另外两套配合机制：
- 工具返回内容过长 → 尝试自动卸载为文件 + 路径引用
- 对话上下文达到阈值 → 旧消息保存到后端，模型输入使用摘要 + 近期消息
~~~

> 上图是为理解而整理的概念图，不代表精确源码调用链。

## 1. 七个内置文件工具

| 工具 | 作用 | 需要记住的细节 |
| --- | --- | --- |
| `ls` | 列举目录 | 查看文件及元信息 |
| `read_file` | 按需读取 | 支持 `offset`、`limit`；文本默认最多前 100 行；支持多模态内容块 |
| `write_file` | 写入文件 | **v0.7 会完整覆盖同路径已有文件** |
| `edit_file` | 精确编辑 | 针对已有内容做字符串替换 |
| `delete` | 删除文件或目录 | v0.7 新增，目录删除应特别审查权限 |
| `glob` | 按文件路径模式查找 | 例如 `**/*.py` |
| `grep` | 按字面量搜索内容 | `files_with_matches` / `content` / `count` 三种输出模式 |

容易误解的地方：

- `read_file(path, offset=100, limit=50)` 从第 101 行读取 50 行（文本场景）；对视频，offset/limit 的单位不同。
- `grep` 和 `glob` 可能返回 `truncated=True`：**工具成功 ≠ 找到了全部结果**。
- 多模态读取还受 Backend 是否提供二进制数据、模型是否支持该模态等因素限制，不能只看扩展名。

## 2. 两种上下文管理机制（不能混为一谈）

### 2.1 大工具结果自动卸载（Eviction）

当适用的普通工具返回内容超过**约 20,000 tokens** 的配置阈值时，文件系统中间件会尝试：

1. 将完整结果写入 Backend。
2. 用**文件路径 + 首尾各 5 行预览**替代对话中的大块工具结果。
3. 如果后续需要更多细节，Agent 再读取文件。

**边界**：该阈值针对适用的工具**返回结果**，不是工具调用参数；`read_file`、`ls`、`glob`、`grep` 等内置文件工具不走这条通用卸载路径。是否成功取决于 Backend 能否写入。

### 2.2 长对话摘要（Summarization）

当上下文达到摘要阈值，Deep Agents 会保存待总结的旧消息，并让模型本轮输入变成**摘要 + 近期消息**；正常摘要时，LangGraph State 中原始消息并不会因此直接消失。

- 模型 profile 提供窗口大小时，`create_deep_agent()` 默认按窗口的 **85%** 触发；否则使用固定阈值，也可自行配置。
- **摘要**负责降低当前模型输入负担；**历史文件**让细节有机会被回查。二者是互补方案。

> 待思考：摘要信息丢失与按需检索遗漏分别会造成什么失败？

## 3. Backend：同一文件接口，不同存储语义

| 后端 | 内容存在哪里 | 适用场景 | 最重要的限制 |
| --- | --- | --- | --- |
| **StateBackend**（默认） | LangGraph Agent State | 会话内草稿、中间结果 | 依赖线程 State；跨调用保留需 Checkpointer + 同一 `thread_id` 或显式传回状态 |
| **FilesystemBackend** | 宿主机本地磁盘 | 本地编程助手、CI | 写入直接落盘；虚拟路径限制不等于沙箱 |
| **LocalShellBackend** | 本地文件 + Shell 执行 | 可信的个人开发环境 | `execute` 在宿主机执行，**没有沙箱隔离** |
| **StoreBackend** | LangGraph Store | 跨线程记忆、用户偏好 | 命名空间决定隔离；`InMemoryStore` 重启进程后会丢数据 |
| **CompositeBackend** | 根据路径路由至多个后端 | 临时文件 + 持久记忆混合 | 路由与各后端的生命周期要分别理解 |
| **沙箱后端** | 隔离执行环境 | 运行代码、处理不可信输入 | 隔离与持久化能力取决于具体沙箱服务 |

### 路径路由举例

~~~text
/workspace/plan.md       → StateBackend：当前任务的临时计划
/memories/preferences.md → StoreBackend：跨线程访问的用户记忆
~~~

这解释了 `CompositeBackend` 的价值：让 Agent 使用一致的文件路径和文件工具，却对不同路径应用不同的存储策略。

**版本提醒**：教程针对 v0.7，`backend=` 应直接传入 Backend 实例，而不是此前的 Backend 工厂函数；但 `StoreBackend(namespace=lambda rt: ...)` 的命名空间函数仍然支持。

## 4. 权限与安全不是 Backend 名称就能保证的

- `FilesystemBackend(root_dir=".", virtual_mode=True)` 的路径约束不等于进程沙箱，也不能代替权限和文件隔离。
- `LocalShellBackend` 的 Shell 命令能访问宿主机资源；把 `root_dir` 设小也不能真正限制 Shell。
- `FilesystemPermission` 支持 `allow` / `deny` / `interrupt`；规则**按顺序匹配，首条命中生效**，未命中默认允许。
- 删除属于文件系统权限中的写入类别；保护敏感路径时必须考虑删除目标本身及其**上级目录**的递归删除风险。
- 定制后端可实现 `BackendProtocol`，用 `GuardedBackend` 或 `PolicyWrapper` 拦截操作；若支持 `delete`，保护逻辑也必须覆盖它。

## 5. 代码实践记录（待实测）

课程建议观察文件系统保存内容和中间结果的过程，可使用官方 [`deepagents/content-builder` 模板](https://github.com/agentseek-ai/agentseek-templates/tree/main/templates/deepagents/content-builder)。

**建议验证的三个小实验：**

- [ ] 用默认 StateBackend 写文件，比较相同/不同 `thread_id` 的可见性。
- [ ] 用 FilesystemBackend 写一个测试文件，检查真实磁盘内容；对比 `write_file` 覆盖与 `edit_file` 替换。
- [ ] 用 CompositeBackend 分别写入 `/workspace/` 与 `/memories/`，观察路径路由和跨线程差异。

**环境、命令、实际输出与报错：** 待实践后填写。不能把预期行为当作已经验证的结果。

## 6. 苏格拉底式问答记录

> 记录原则：每轮只问一个问题。保留你的原始观点、理解情况、追问与经核对的补充；不会将 AI 的补充冒充成你的回答。

### Q1｜为什么上下文窗口足够大，仍需要文件系统？

**问题**：假设研究 Agent 搜索到了 5 万 token 的资料，而模型支持 128K 上下文。为什么 Deep Agents 仍倾向把资料保存成文件、需要时再读，而不是把全部资料长期留在对话中？

- **我的回答**：待填写
- **理解评估**：待回答后评估
- **追问 / 更正**：待填写
- **知识补充**：待回答后补充

### 后续问题方向（不是现在的作业）

1. StateBackend 的“临时”究竟以什么为边界？
2. `CompositeBackend` 如何将草稿和长期记忆分开？
3. 路径约束、权限拦截与进程沙箱有什么区别？
4. 自动摘要和自动卸载分别可能丢失什么信息？
5. 如何用最小实验验证这些结论？

## 7. 阶段性总结

**课程已经明确的结论：**

1. 虚拟文件系统是 Deep Agents 上下文工程的一部分，不只是存储 API。
2. 文件工具提供操作能力；Backend 决定存储和生命周期；权限决定操作能否执行。
3. 自动卸载与自动摘要都试图降低模型本轮输入压力，但触发条件、保留形式不同。

**我的理解（随学习持续更新）：** 等待 Q1 回答。  
**仍有疑问：** 等待问答与实验。  
**完成状态：** [x] 整理课程知识框架　[ ] 苏格拉底问答　[ ] 实验验证　[ ] 个人总结

## 参考资料

- [DeepAgents In Action 第三章原文](https://github.com/datawhalechina/deepagents-in-action/blob/main/content/ch03-virtual-filesystem.md)
- [课程在线阅读：第三章](https://datawhalechina.github.io/deepagents-in-action/chapters/ch03-virtual-filesystem/)
- [Deep Agents 官方文档](https://docs.langchain.com/oss/python/deepagents/overview)

---

*本笔记为个人学习整理。知识框架依据课程原文，问答中的个人观点将根据实际回答持续补充。*
