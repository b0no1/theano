# Random tips

## Integer division

```python
float(6)/float(5)  # 1.2
float(6)//float(5) # 1.0
```

## a2string

```python
print "what a ni" + str(8) # what a ni8
```

### ++ doesn't work

```python
a = 2
a += 2
print a
```

## Raw strings

```python
s=r'no escape \n\t'
print s
```

### End with \\ to continue the line

>  A method is like a function, but it runs "on" an object.

## Common str methods

```python
s.strip()
s.isalpha(), s.isdigit(), s.isspace()
s.startswith('somestring'), s.endswith('other')
s.split('delim')
s.join(list)
```

> Python does not have a separate scalar "char" type.

### Conservation of characters

```python
s[:n] + s[n:] == s
```

## Print format string

*print* takes a string with % operators as format, to print the tuple on the right.

```python
a=(1,2,3) # a tuple : items separated by ,
print '%d %d %d' %a # prints 1 2 3
print '%d %d %d' %(1,2,3) # prints 1 2 3
```

### Code across lines

```python
text = ("%d little pigs come out or I'll %s and %s and %s" %
    (3, 'huff', 'puff', 'blow down'))
```

### new_variable = list_name does not copy the list

## for/in and if/in

```python
mylist=[1,'one',2,'two']
concat=''
for num in mylist:
	concat += str(num)
print concat

x = 2
if x in mylist:
	print 'x is present in mylist'
```

## List Methods

```python
list.insert(index,item)
list1.extend(list2)
list.index(item), list.remove(item)
list.pop(index), list.pop() # remove last element
```

### List replace

```python
list[0:2] = 'z'    ## replace ['a', 'b'] with ['z']
```