import dataclasses


# @dataclasses.dataclass
# class Test:
#     test1: int
#     test2: int

#     def __iter__(self):
#         for field in dataclasses.fields(self):
#             yield getattr(self, field.name)


# test_list = [Test(1, 2), Test(2, 3), Test(3, 4)]

# for var1, var2 in test_list:
#     print(var1, var2)


test_var = 0

def func():
    global test_var

    print(test_var)
    test_var = "xd"
    print(test_var)

print(test_var)
func()
print(test_var)