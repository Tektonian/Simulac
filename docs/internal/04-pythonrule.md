# Follow google

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