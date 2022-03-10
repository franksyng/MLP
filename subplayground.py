# list = [[[321, 329], 6], [[404, 413], 6], [[652, 661], 6], [[26, 38], 9], [[96, 118], 9], [[[56, 69]], 10], [[[5, 9]], 11], [[[10, 11]], 12]]
# # list = [sorted(each, key=lambda x:x[0]) for each in list]
# s_list = sorted(list, key=lambda x:x[0][0], reverse=True)
#
s = "Hello there!"
a = [0, 9]
b = [2, 4]
# print(s[0:6])
left = s[:a[0]]
mid = s[a[0]:a[1]]
right = s[a[1]:]
# s_list = ['O']  * 10
final = []
final.append(right)
for each in mid.split(' ')[::-1]:
    final.append(each)
print(final[::-1])