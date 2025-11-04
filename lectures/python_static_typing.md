---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(python_static_typing)=
```{raw} jupyter
<div id="qe-notebook-header" align="right" style="text-align:right;">
        <a href="https://quantecon.org/" title="quantecon.org">
                <img style="width:250px;display:inline;" width="250px" src="https://assets.quantecon.org/img/qe-menubar-logo.svg" alt="QuantEcon">
        </a>
</div>
```

# Python Static Typing

```{index} single: Python; Type Hints
```

```{index} single: Python; Static Typing
```

## Overview

Python is a dynamically typed language, which means that variable types are determined at runtime rather than compile time.

While this flexibility is one of Python's great strengths, it can sometimes lead to runtime errors that could be caught earlier with explicit type information.

Python 3.5 introduced **type hints** (also called type annotations), which allow developers to specify expected types for variables, function parameters, and return values.

Type hints have several important applications:

1. **Improving JIT compiler efficiency** - while Numba doesn't currently use them, future JIT compilers may leverage type information for better optimization
2. **Better software design** - explicit types serve as documentation and help define clear interfaces
3. **LLM code generation** - modern AI tools often generate code with type hints, making familiarity important
4. **Static analysis and error detection** - tools like `mypy` and `pyright` can catch type-related errors before runtime

This lecture will introduce you to Python's type annotation syntax and show how to use type hints effectively in your code.

```{note}
Type hints in Python are optional and do not affect runtime behavior. They are primarily used for documentation, static analysis, and tooling support.
```

## Basic Type Annotations

### Variable Annotations

The simplest form of type annotation is for variables:

```{code-cell} python3
# Basic type annotations
name: str = "Alice"
age: int = 30
salary: float = 75000.50
is_employed: bool = True

print(f"{name} is {age} years old")
```

You can also annotate variables without immediate assignment:

```{code-cell} python3
# Forward declaration with type annotation
count: int
data: list

count = 10
data = [1, 2, 3, 4, 5]
print(f"Count: {count}, Data: {data}")
```

### Function Annotations

Type hints are most commonly used with functions to specify parameter types and return types:

```{code-cell} python3
def greet(name: str, age: int) -> str:
    """Return a greeting message."""
    return f"Hello {name}, you are {age} years old!"

def calculate_tax(income: float, rate: float) -> float:
    """Calculate tax based on income and rate."""
    return income * rate

# Using the functions
message = greet("Bob", 25)
tax = calculate_tax(50000.0, 0.2)

print(message)
print(f"Tax: ${tax:.2f}")
```

### Collection Types

For collections, you need to import types from the `typing` module (Python 3.9+ also supports built-in generics):

```{code-cell} python3
from typing import List, Dict, Tuple, Set, Optional

def process_scores(scores: List[float]) -> Dict[str, float]:
    """Process a list of scores and return statistics."""
    return {
        'mean': sum(scores) / len(scores),
        'max': max(scores),
        'min': min(scores)
    }

def get_coordinates() -> Tuple[float, float]:
    """Return x, y coordinates."""
    return (3.14, 2.71)

def find_unique_words(text: str) -> Set[str]:
    """Return unique words from text."""
    return set(text.lower().split())

# Example usage
test_scores = [85.5, 92.0, 78.3, 95.7, 88.1]
stats = process_scores(test_scores)
coords = get_coordinates()
words = find_unique_words("The quick brown fox jumps over the lazy dog")

print(f"Score statistics: {stats}")
print(f"Coordinates: {coords}")
print(f"Unique words: {len(words)} words")
```

### Optional and Union Types

Use `Optional` for values that might be `None`, and `Union` for values that could be one of several types:

```{code-cell} python3
from typing import Optional, Union

def divide(a: float, b: float) -> Optional[float]:
    """Divide two numbers, return None if division by zero."""
    if b == 0:
        return None
    return a / b

def process_id(user_id: Union[int, str]) -> str:
    """Process user ID, which can be either int or string."""
    return f"User ID: {user_id}"

# Example usage
result1 = divide(10, 2)
result2 = divide(10, 0)
print(f"10/2 = {result1}")
print(f"10/0 = {result2}")

print(process_id(123))
print(process_id("abc123"))
```

## Advanced Type Features

### Generic Types

Generics allow you to write functions and classes that work with multiple types:

```{code-cell} python3
from typing import TypeVar, Generic, List

T = TypeVar('T')

def get_first_item(items: List[T]) -> Optional[T]:
    """Get the first item from a list."""
    if items:
        return items[0]
    return None

def reverse_list(items: List[T]) -> List[T]:
    """Reverse a list maintaining the same type."""
    return items[::-1]

# Example usage with different types
numbers = [1, 2, 3, 4, 5]
words = ["apple", "banana", "cherry"]

first_number = get_first_item(numbers)  # Type is Optional[int]
first_word = get_first_item(words)      # Type is Optional[str]

reversed_numbers = reverse_list(numbers)  # Type is List[int]
reversed_words = reverse_list(words)      # Type is List[str]

print(f"First number: {first_number}")
print(f"First word: {first_word}")
print(f"Reversed numbers: {reversed_numbers}")
print(f"Reversed words: {reversed_words}")
```

### Custom Classes with Type Hints

Type hints work well with custom classes:

```{code-cell} python3
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Point:
    """A point in 2D space."""
    x: float
    y: float
    
    def distance_from_origin(self) -> float:
        """Calculate distance from origin."""
        return (self.x**2 + self.y**2)**0.5

class Portfolio:
    """A simple investment portfolio."""
    
    def __init__(self, name: str) -> None:
        self.name = name
        self.holdings: Dict[str, float] = {}
    
    def add_stock(self, symbol: str, shares: float) -> None:
        """Add shares of a stock to the portfolio."""
        if symbol in self.holdings:
            self.holdings[symbol] += shares
        else:
            self.holdings[symbol] = shares
    
    def get_holding(self, symbol: str) -> Optional[float]:
        """Get number of shares for a given stock."""
        return self.holdings.get(symbol)
    
    def total_positions(self) -> int:
        """Return total number of different stock positions."""
        return len(self.holdings)

# Example usage
point = Point(3.0, 4.0)
print(f"Distance from origin: {point.distance_from_origin():.2f}")

portfolio = Portfolio("My Portfolio")
portfolio.add_stock("AAPL", 100)
portfolio.add_stock("GOOGL", 50)
print(f"AAPL holding: {portfolio.get_holding('AAPL')}")
print(f"Total positions: {portfolio.total_positions()}")
```

## Type Checking with mypy

While Python doesn't enforce type hints at runtime, you can use static type checkers like `mypy` to validate your code:

```{code-cell} python3
# This function has a type error that mypy would catch
def add_numbers(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

# This would cause a type error in mypy:
# result = add_numbers("hello", 5)  # Error: str is not compatible with int

# Correct usage:
result = add_numbers(10, 20)
print(f"Result: {result}")
```

To check your code with mypy, you would run:
```bash
pip install mypy
mypy your_file.py
```

## Applications in Scientific Computing

### NumPy Arrays

Type hints work well with NumPy arrays:

```{code-cell} python3
import numpy as np
from typing import Union

# Type alias for NumPy arrays
FloatArray = np.ndarray

def normalize_vector(vec: FloatArray) -> FloatArray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def compute_statistics(data: FloatArray) -> Dict[str, float]:
    """Compute basic statistics for an array."""
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data))
    }

# Example usage
vector = np.array([3.0, 4.0])
normalized = normalize_vector(vector)
print(f"Original: {vector}, Normalized: {normalized}")

data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
stats = compute_statistics(data)
print(f"Statistics: {stats}")
```

### Economic Modeling Example

Here's a more comprehensive example showing type hints in an economic context:

```{code-cell} python3
from typing import NamedTuple, List, Callable
import numpy as np

class EconomicParameters(NamedTuple):
    """Parameters for a simple economic model."""
    alpha: float  # Capital share
    beta: float   # Discount factor
    delta: float  # Depreciation rate
    
class ModelResult(NamedTuple):
    """Results from economic model simulation."""
    capital: np.ndarray
    consumption: np.ndarray
    utility: float

def utility_function(consumption: float, gamma: float = 2.0) -> float:
    """CRRA utility function."""
    if gamma == 1.0:
        return np.log(consumption)
    else:
        return (consumption**(1 - gamma) - 1) / (1 - gamma)

def simulate_growth_model(
    params: EconomicParameters,
    k0: float,
    periods: int = 100
) -> ModelResult:
    """Simulate a simple growth model."""
    capital = np.zeros(periods + 1)
    consumption = np.zeros(periods)
    capital[0] = k0
    
    for t in range(periods):
        # Simple optimal policy (not solving the full problem)
        investment_rate = 0.3
        output = capital[t] ** params.alpha
        investment = investment_rate * output
        consumption[t] = output - investment
        capital[t + 1] = (1 - params.delta) * capital[t] + investment
    
    # Calculate total discounted utility
    total_utility = sum(
        (params.beta ** t) * utility_function(consumption[t])
        for t in range(periods)
    )
    
    return ModelResult(capital[:-1], consumption, total_utility)

# Example usage
params = EconomicParameters(alpha=0.33, beta=0.95, delta=0.1)
result = simulate_growth_model(params, k0=1.0, periods=50)

print(f"Final capital: {result.capital[-1]:.3f}")
print(f"Average consumption: {np.mean(result.consumption):.3f}")
print(f"Total utility: {result.utility:.3f}")
```

## Limitations and Considerations

### Numba Compatibility

Currently, Numba does not use Python type hints for compilation. You still need to use Numba's decorator syntax:

```{code-cell} python3
import numba as nb

# Type hints are ignored by Numba
@nb.jit
def fast_sum(arr: np.ndarray) -> float:
    """Fast sum using Numba - type hints ignored."""
    total = 0.0
    for i in range(len(arr)):
        total += arr[i]
    return total

# Numba-specific type specification (current approach)
@nb.jit(nb.float64(nb.float64[:]))
def fast_sum_numba(arr):
    """Fast sum with Numba type specification."""
    total = 0.0
    for i in range(len(arr)):
        total += arr[i]
    return total

# Test both functions
test_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
print(f"Sum with type hints: {fast_sum(test_array)}")
print(f"Sum with Numba types: {fast_sum_numba(test_array)}")
```

```{note}
According to the [Numba documentation](https://stackoverflow.com/questions/42310278/using-python-type-hints-with-numba), type hints are not yet supported for JIT compilation. However, this may change in future versions.
```

### JAX Type Annotations

JAX has a [roadmap for supporting Python type annotations](https://docs.jax.dev/en/latest/jep/12049-type-annotations.html), which will eventually provide better integration:

```{code-cell} python3
# This is more of a conceptual example
# JAX type annotation support is still in development

import jax.numpy as jnp

def jax_function(x: jnp.ndarray) -> jnp.ndarray:
    """JAX function with type hints."""
    return jnp.sin(x) + jnp.cos(x**2)

# Example usage
x = jnp.array([1.0, 2.0, 3.0])
result = jax_function(x)
print(f"JAX result: {result}")
```

## Best Practices

1. **Start gradually**: Add type hints to new code and gradually retrofit existing code
2. **Focus on interfaces**: Prioritize type hints for function signatures and public APIs
3. **Use type aliases**: Create readable aliases for complex types
4. **Be consistent**: Maintain consistent typing style across your codebase
5. **Leverage tools**: Use mypy or pyright for static type checking
6. **Document with types**: Let type hints serve as part of your documentation

```{code-cell} python3
# Example of good type hint practices
from typing import List, Dict, TypeAlias

# Type aliases for clarity
Price = float
Quantity = int
StockData = Dict[str, Price]

def calculate_portfolio_value(
    holdings: Dict[str, Quantity],
    prices: StockData
) -> Price:
    """Calculate total portfolio value.
    
    Args:
        holdings: Dictionary mapping stock symbols to quantities
        prices: Dictionary mapping stock symbols to current prices
        
    Returns:
        Total portfolio value
    """
    total_value = 0.0
    for symbol, quantity in holdings.items():
        if symbol in prices:
            total_value += quantity * prices[symbol]
    return total_value

# Example usage
my_holdings = {"AAPL": 100, "GOOGL": 50}
current_prices = {"AAPL": 150.0, "GOOGL": 2500.0}

portfolio_value = calculate_portfolio_value(my_holdings, current_prices)
print(f"Portfolio value: ${portfolio_value:,.2f}")
```

## Tools and Integration

### Modern Type Checking Tools

1. **mypy**: The original Python type checker
2. **pyright/pylance**: Microsoft's type checker used in VS Code
3. **pyre**: Facebook's type checker for large codebases

### IDE Integration

Most modern IDEs provide excellent support for type hints:

- **VS Code**: Built-in support with Pylance extension
- **PyCharm**: Native type checking and completion
- **Vim/Neovim**: Via language server protocol (LSP)

## Exercises

### Exercise 1

Write a function `calculate_compound_interest` with proper type hints that:
- Takes principal amount, annual interest rate, and number of years
- Returns the final amount after compound interest
- Include proper type annotations for all parameters and return value

```{code-cell} python3
def calculate_compound_interest(principal: float, rate: float, years: int) -> float:
    """Calculate compound interest.
    
    Args:
        principal: Initial amount invested
        rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
        years: Number of years
        
    Returns:
        Final amount after compound interest
    """
    return principal * (1 + rate) ** years

# Test the function
result = calculate_compound_interest(1000.0, 0.05, 10)
print(f"$1000 at 5% for 10 years: ${result:.2f}")
```

### Exercise 2

Create a class `BankAccount` with type hints that:
- Stores account holder name and balance
- Has methods to deposit, withdraw, and check balance
- Uses appropriate type hints throughout

```{code-cell} python3
from typing import Optional

class BankAccount:
    """A simple bank account with type annotations."""
    
    def __init__(self, account_holder: str, initial_balance: float = 0.0) -> None:
        self.account_holder = account_holder
        self.balance = initial_balance
    
    def deposit(self, amount: float) -> None:
        """Deposit money into the account."""
        if amount > 0:
            self.balance += amount
    
    def withdraw(self, amount: float) -> bool:
        """Withdraw money from account. Returns True if successful."""
        if 0 < amount <= self.balance:
            self.balance -= amount
            return True
        return False
    
    def get_balance(self) -> float:
        """Get current account balance."""
        return self.balance
    
    def __str__(self) -> str:
        return f"Account({self.account_holder}): ${self.balance:.2f}"

# Test the class
account = BankAccount("Alice", 1000.0)
account.deposit(500.0)
success = account.withdraw(200.0)
print(f"Withdrawal successful: {success}")
print(account)
```

### Exercise 3

Write a function that processes economic time series data with proper type annotations:

```{code-cell} python3
from typing import Tuple
import numpy as np

def analyze_time_series(
    data: np.ndarray,
    window_size: int = 5
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Analyze time series data.
    
    Args:
        data: Time series data as numpy array
        window_size: Size of moving average window
        
    Returns:
        Tuple of (moving_averages, differences, volatility)
    """
    # Calculate moving averages
    moving_averages = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    # Calculate first differences
    differences = np.diff(data)
    
    # Calculate volatility (standard deviation of differences)
    volatility = float(np.std(differences))
    
    return moving_averages, differences, volatility

# Test with sample data
sample_data = np.array([100, 102, 98, 105, 103, 107, 104, 109, 106, 111])
ma, diffs, vol = analyze_time_series(sample_data, window_size=3)

print(f"Moving averages: {ma}")
print(f"First differences: {diffs}")
print(f"Volatility: {vol:.2f}")
```

## Summary

Python type hints provide a powerful way to make your code more readable, maintainable, and less error-prone. While they don't affect runtime behavior, they offer significant benefits:

- **Documentation**: Type hints serve as inline documentation for your code
- **IDE Support**: Better autocomplete, refactoring, and error detection
- **Static Analysis**: Catch errors before runtime using tools like mypy
- **Team Collaboration**: Clearer interfaces make code easier to understand and maintain

As the Python ecosystem evolves, type hints are becoming increasingly important, especially with the rise of AI-generated code and more sophisticated development tools.

Key takeaways:
- Start with function signatures and gradually add more detailed typing
- Use type checkers like mypy to validate your annotations
- Consider type hints as part of your code documentation strategy
- Be aware of current limitations with performance libraries like Numba
- Stay informed about developments in JAX and other scientific computing libraries

```{note}
For more information on Python typing, see the [official documentation](https://docs.python.org/3/library/typing.html) and the [mypy documentation](https://mypy.readthedocs.io/).
```