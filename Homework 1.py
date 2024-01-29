from collections import Counter

### Please fill in an explanation of each function and an example of how to use it below.

## clear() - Removes every item in the counter, returning an empty counter
# Creating a list of digits to count
list1 = [1,1,1,1,4,6,7,7,7,6,3,4,8,9,0,7,6,9,9,9,6,3,2,0]
list1

# Check how many of each digit are in list1 using Counter()
counter1 = Counter(list1)
counter1

# clear the counter using the .clear() function
counter1.clear()
counter1

## copy() - Copies every item in the counter and returns each copied element in dict form
# Start with list1 and use the counter function
list1
counter1 = Counter(list1)
counter1

# Use the copy function to create a copy of the first counter into a new counter
counter_copy = counter1.copy()
counter_copy

## 	elements() - Returns a data structure composed of several instances of each unique value in the counter.
## The number of instances of each value is equal to their value count
# Start with list1 and use counter
list1
counter1 = Counter(list1)
counter1

# use the elements() funtion to get the elements in the counter
# Use a for loop to print all the elements in the counter
elements_in_counter = counter1.elements()
for i in elements_in_counter:
    print(i)

### get() - Retrieves the value corresponding to the specified key
# Start with list1 and use the counter
list1
counter1 = Counter(list1)
counter1
# use the get() function to see how many elements are in the counter
get_1_count = counter1.get(1)
get_1_count
print("there are", get_1_count, "counts of 1 in list1")

### items() - Returns all key/value pairs in a counter
# using counter1, find the key/value pairs using items()
counter1
items_in_counter1 = counter1.items()
items_in_counter1
print("The Key/value pairs in counter1 are", items_in_counter1)

### most_common() - Returns a list of tuples where each tuple contains both the key and the value of the counter.
### The list is sorted in descending order based on the value of the key
# using counter1, create a sorted list of the value of key
counter1
most_common_in_list1 = counter1.most_common()
most_common_in_list1

### pop() - Removes the key-value pair corresponding to the input key and returns the corresponding value
# using count1, pop a key and return the value
counter1
key_value_for_7 = counter1.pop(7)
key_value_for_7

###	setdefault() - Returns the value corresponding with the input key. 
### If the key does not exist, insert the key into the Counter and 
### assign it a value corresponding with the second method input
# using counter1, set a new default value for a key and insert a new key with a defauly value
counter1
value_9 = counter1.setdefault(9)
value_A = counter1.setdefault('A', 60)

# since A was not in the originally in the counter, it is added with a value of 60
print("The value for 9 is", value_9)
print("The value for new key A is", value_A)
print("the updated counter including A is", counter1)