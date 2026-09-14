# Developer Contribution Guide

Welcome to contribute to the MindSpeed project! This guide is intended to help developers understand how to contribute to the project, including the development workflow, coding standards, testing requirements, and more.

---

## 1. Contribution Process

### 1.1 Feature Proposal and Review

Before starting development of a new feature, follow the process below:

1. **Submit an issue**: Submit an issue on GitCode describing the feature to be implemented or the problem to be fixed.
2. **Design review**:
   - Huawei employees: Search for `MindSpeed-Core PR Review Group` on WeLink to conduct the design review
   - Community contributors: Conduct public discussion through issues
3. **Solution design**: Write a software solution design document, using software engineering languages such as class diagrams and sequence diagrams to describe the overall solution and modification points. For design specifications, refer to [MindSpeed Core design document](https://gitcode.com/Ascend/MindSpeed/wiki/MindSpeedCore%E8%AE%BE%E8%AE%A1%E6%96%87%E6%A1%A3.md).
4. **Read the development specifications**: Carefully read the document, comply with the development specifications, avoid introducing technical debt, and facilitate subsequent maintenance.

### 1.2 Development Process

```plaintext
Feature Proposal → Design Review → Development → Testing → Documentation → PR Submission → CI Verification → Code Review → Merge
```

---

## 2. Code Development Standards

### 2.1 Feature Registration

- A new feature must be registered by inheriting the `MindSpeedFeature` class. For details, see [Feature Development](docs/en/user-guide/feature_development.md).
- Create folders by feature dimension under the `mindspeed/features_manager/` directory.
- Feature files should be self-contained and functionally cohesive.

### 2.2 Patch Enabling Rules

- **Prohibit enabling by default**: A new feature must not enable related patches by default (except for native feature adaptation).
- **Patch registration**: Register patches through the `MindSpeedPatchesManager.register_patch` function.
- **Monkey patch usage restrictions**: Avoid using Monkey patch whenever possible, and use it only when no other solution is available.

### 2.3 Code Design Principles

1. **Architectural decoupling**: The feature itself and the framework adaptation should be decoupled to facilitate adaptation and enabling across different frameworks.
2. **Decorator usage**: When using decorators, ensure that the original function is always executed and its original semantics are not changed.
3. **Class replacement preferred**: If both `__init__` and `forward` need to be patched, it is recommended to create a new class for replacement.
4. **Factory pattern**: When incompatible features modify the same class, consider using the factory pattern for implementation through configuration.
5. **Abstraction**: Use abstraction, extraction, and relocation techniques to anticipate future version iterations and avoid code rot.
6. **Accurate naming**: Pay attention to the accuracy of function/variable naming, and avoid names with ambiguous expressions.

### 2.4 Code Quality Requirements

| Metric | Limit |
|------|------|
| Lines per function | ≤ 200 lines |
| Lines per file | ≤ 1000 lines |
| Naming convention | Clear and accurate; avoid ambiguous naming |
| God class/function | Prohibited |

### 2.5 Compatibility Handling

- When incompatibility with existing features is discovered, add code interception and error reporting, and add compatibility notes to the documentation

---

## 3. Test Case Specifications

### 3.1 Test Case Location

For a new feature, create test cases in an appropriate location under the `tests_extend/unit_tests` directory based on the feature dimension.

### 3.2 Test Requirements

1. **Scenario coverage**: Ensure that test cases cover the main usage scenarios.
2. **Concise and complete**: Test cases should be concise yet cover the necessary scenarios; more is not necessarily better.
3. **Parameterized tests**: Use `pytest.mark.parametrize` to test multiple data types and input parameters.

### 3.3 Distributed Test Cases

- Distributed test cases can inherit from the `unit_tests.common.DistributedTest` class.
- Distributed initialization is time-consuming. Do not inherit from `DistributedTest` for single-device test cases.

---

## 4. Documentation Guidelines

### 4.1 Document Location

A new feature must store its documentation under the `docs` folder at a path selected according to the feature dimension.

### 4.2 Documentation Content Requirements

The documentation must include the following content:

- **Problem Analysis**: Clearly describe the problem being solved.
- **Solution**: Explain the technical solution in detail; do not summarize it in one or two sentences.
- **Usage Scenarios**: Include usage restrictions and applicable conditions.
- **Usage**: Provide code examples and configuration instructions.
- **Effect**: Present performance or accuracy improvement data.

### 4.3 Document Format

- It is recommended to use software engineering languages such as class diagrams, activity diagrams, and sequence diagrams to describe the solution
- If necessary, add compatibility notes between features

---

## 5. Code Submission and PR Specifications

### 5.1 PR Title and Description

- **Title**: Concise and complete, summarizing the main modifications.
- **Description**: Clearly state the purpose, content, and impact of the modifications.
- **Test results**: For a new feature, provide accuracy and performance test results.

### 5.2 CI Gate

- After submitting a PR, comment `compile` to trigger the CI gate check.
- Merging requires passing the gate and review by at least two developers (+2).

### 5.3 Branch Management

- Consider the branches involved; if necessary, submit PRs to both master and the corresponding branch
- Pay attention to the release cadence and reserve time in advance for PR revisions and merging; requests to merge first and revise later will not be accepted

### 5.4 Code Check

- Follow the coding standards. Code check suppressions must be reviewed line by line. Batch or arbitrary suppression requests are prohibited.
- Commits exceeding 500 lines require a code walkthrough and review.
- Strictly enforce quality. Avoid low-level mistakes, and resolve issues as soon as they are identified.

### 5.5 Open Source Compliance

- Check the open source license of the introduced open source code.
- Add the corresponding copyright notice to new files.

### 5.6 Commit Strategy

- **Commit often, but in small increments**: Avoid submitting a large amount of code at once.
- **Quality first**: Ensure code quality in every commit.

---

## 6. Community Code of Conduct

### 6.1 Contributor Conventions

- Respect the work and opinions of others.
- Maintain friendly and professional communication.
- Follow the project's code standards and best practices.
- Respond to feedback and issues in a timely manner.

### 6.2 Issue Feedback

- Use clear titles and detailed descriptions.
- Provide reproduction steps and environment information.
- Follow the issue template requirements.

---

## 7. Contact Us

- **GitHub/GitCode**: Communicate through issues and PRs.
- **Huawei employees**: Search for `MindSpeed TMG group` on WeLink.
- **Community discussions**: Join the project's Discussions section.

---

Thank you to all developers for your contributions! The development of MindSpeed depends on the efforts of every developer.

---

*MindSpeed Team*
