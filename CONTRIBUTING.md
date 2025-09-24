# Contributing to Enhanced Safety Patrol Bot

Thank you for your interest in contributing to the Enhanced Safety Patrol Bot project! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Reporting Issues
- Use the GitHub issue tracker to report bugs or request features
- Provide detailed information about the issue, including:
  - Steps to reproduce
  - Expected vs actual behavior
  - System information (OS, Python version, Webots version)
  - Screenshots or error messages when applicable

### Suggesting Enhancements
- Open an issue with the "enhancement" label
- Describe the proposed feature in detail
- Explain the use case and benefits
- Consider implementation complexity

### Code Contributions

1. **Fork the repository**
   ```bash
   git clone https://github.com/phanindhar157/enhanced-safety-patrol-bot.git
   cd enhanced-safety-patrol-bot
   ```

2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make your changes**
   - Follow the existing code style
   - Add comments for complex logic
   - Update documentation as needed
   - Test your changes thoroughly

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "Add: Brief description of your changes"
   ```

5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create a Pull Request**
   - Provide a clear title and description
   - Reference any related issues
   - Include screenshots for UI changes

## 📋 Development Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use meaningful variable and function names
- Add docstrings for functions and classes
- Keep functions focused and modular

### Testing
- Test your changes with different scenarios
- Ensure compatibility with Webots R2023b+
- Verify AI model functionality
- Test web interface components

### Documentation
- Update README.md for significant changes
- Add inline comments for complex algorithms
- Update API documentation if applicable
- Include usage examples

## 🧠 AI Model Contributions

### Adding New Models
- Follow the existing model structure in `ml_models/`
- Include training scripts and evaluation metrics
- Provide sample datasets or data generation methods
- Document model performance and limitations

### Dataset Contributions
- Ensure datasets are properly formatted
- Include metadata and documentation
- Follow the existing dataset structure
- Consider data privacy and licensing

## 🚨 Safety Considerations

This project involves safety-critical systems. Please ensure:
- All safety features are thoroughly tested
- Emergency response systems are validated
- AI model predictions are reliable
- Documentation includes safety warnings

## 📝 Commit Message Guidelines

Use clear, descriptive commit messages:
- `Add:` for new features
- `Fix:` for bug fixes
- `Update:` for improvements
- `Remove:` for deletions
- `Docs:` for documentation changes

Examples:
- `Add: Fire detection model with thermal imaging`
- `Fix: Gas sensor calibration issue`
- `Update: Web dashboard with real-time alerts`
- `Docs: Add installation guide for Windows`

## 🔍 Review Process

All contributions will be reviewed for:
- Code quality and style
- Functionality and testing
- Documentation completeness
- Safety considerations
- Performance impact

## 📞 Getting Help

- Check existing issues and discussions
- Join our community discussions
- Contact maintainers for complex questions

## 🎉 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to making industrial safety more intelligent and automated!
