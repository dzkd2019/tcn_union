# Engineering Principles

## 1. Simplicity First

- Always prefer the simplest correct solution that satisfies the current requirements.
- Optimize for readability, maintainability, and low cognitive complexity, not architectural sophistication.
- Prefer explicit and straightforward code over clever abstractions.

## 2. KISS and YAGNI

- Do not implement features that were not explicitly requested.
- Do not introduce speculative abstractions for hypothetical future requirements.
- Do not create interfaces, factories, strategies, providers, registries, or plugin systems when a direct implementation is sufficient.
- Do not add configuration options unless there is a real requirement for configurability.
- Do not introduce additional dependencies when existing tools or standard libraries are sufficient.

## 3. Architecture Discipline

- Reuse existing project architecture and conventions whenever possible.
- Do not introduce new architectural layers without a concrete justification.
- Avoid unnecessary wrappers, adapters, managers, services, and pass-through functions.
- Keep related logic together when separation would increase complexity.
- Introduce abstractions only when they provide demonstrable value, such as eliminating meaningful duplication or isolating a genuinely independent responsibility.

## 4. Implementation Discipline

- Inspect existing code before implementing new functionality.
- Prefer modifying existing code over creating new files and modules.
- Keep changes narrowly scoped to the requested task.
- Do not refactor unrelated code.
- Do not add defensive checks for impossible scenarios.
- Do not overengineer error handling.

## 5. Before Adding Complexity

Before introducing a new abstraction, dependency, module, or architectural pattern, ask:

1. Does the current requirement actually need this?
2. Can the requirement be solved using existing code?
3. Would a straightforward implementation be easier to understand?
4. Does the additional complexity provide a concrete benefit today?

If the simpler solution is sufficient, use it.

## 6. Completion Review

Before completing a task:

- Review the entire diff.
- Identify unnecessary abstractions, duplicated logic, speculative features, and redundant code.
- Simplify the implementation where appropriate.
- Ensure existing behavior is preserved unless a behavior change was requested.
- Run relevant tests and report their results honestly.

## 7. Design Justification

When introducing a new architectural layer or significant abstraction, briefly explain:

- What problem it solves.
- Why existing code cannot solve it adequately.
- Why a simpler alternative is insufficient.

Do not introduce complexity merely to demonstrate good software architecture.

## 8. 图表输出规范

- 绘图时同时输出 **PNG** 和 **SVG** 两种格式；
- 不再输出 PDF 图片；
- PNG 用于 PPT、网页和快速预览，SVG 用于论文/报告的矢量插图；
- 文件名保持一致，仅扩展名不同，例如 `fig_name.png` 与 `fig_name.svg`。
- 如果旧脚本仍输出 PDF，修改该脚本时一并改为 PNG + SVG。
