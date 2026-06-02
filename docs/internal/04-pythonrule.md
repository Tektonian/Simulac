# Basic rule

1. Follow google python style guide
2. For other cases, follow the rules below

# Follow google

- https://google.github.io/styleguide/pyguide.html

# Comment rule

```py
❌
class Foo:
    # Flags below should be removed
    related_flag1
    related_flag2
    nonrelated_flat1

✅
class Bar:
    # Flags below should be removed
    related_flag1
    related_flag2

    nonrelated_flat1
✅
class FooFoo:
    # region Flags below should be removed
    related_flag1
    related_flag2
    # end-region
    nonrelated_flat1
```

# Import rule
Display lazy import with comment
```py
# import super_big_package - lazy import
```

# Explicit if-else

❌
```python
foo_or_bar: Literal["foo", "bar"] = "foo"

if foo_or_bar == "foo":
    print("FOOOO?")
else:
    print("BARRR!!")
```

✅
```python
foo_or_bar: Literal["foo", "bar"] = "foo"

if foo_or_bar == "foo":
    print("FOOOO?")
else if foo_or_bar == "bar":
    print("BARRR!!")
```