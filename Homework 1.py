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

from itertools import *
flower_orders=['W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y','B/Y','B/Y','B/Y','B/Y','B/Y','R/B/Y','R/B/Y','R/B/Y','R/B/Y','R/B/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/G','W/G','W/G','W/G','R/Y','R/Y','R/Y','R/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y','W/R/B/V','W/R/B/V','W/R/B/V','W/R/B/V','W/N/R/V/Y','W/N/R/V/Y','W/N/R/V/Y','W/N/R/V/Y','N/R/Y','N/R/Y','N/R/Y','W/V/O','W/V/O','W/V/O','W/N/R/Y','W/N/R/Y','W/N/R/Y','R/B/V/Y','R/B/V/Y','R/B/V/Y','W/R/V/Y','W/R/V/Y','W/R/V/Y','W/R/B/V/Y','W/R/B/V/Y','W/R/B/V/Y','W/N/R/B/Y','W/N/R/B/Y','W/N/R/B/Y','R/G','R/G','B/V/Y','B/V/Y','N/B/Y','N/B/Y','W/B/Y','W/B/Y','W/N/B','W/N/B','W/N/R','W/N/R','W/N/B/Y','W/N/B/Y','W/B/V/Y','W/B/V/Y','W/N/R/B/V/Y/G/M','W/N/R/B/V/Y/G/M','B/R','N/R','V/Y','V','N/R/V','N/V/Y','R/B/O','W/B/V','W/V/Y','W/N/R/B','W/N/R/O','W/N/R/G','W/N/V/Y','W/N/Y/M','N/R/B/Y','N/B/V/Y','R/V/Y/O','W/B/V/M','W/B/V/O','N/R/B/Y/M','N/R/V/O/M','W/N/R/Y/G','N/R/B/V/Y','W/R/B/V/Y/P','W/N/R/B/Y/G','W/N/R/B/V/O/M','W/N/R/B/V/Y/M','W/N/B/V/Y/G/M','W/N/B/V/V/Y/P']

### 1. Build your own counter object, then use the built-in Counter() and confirm they have the same values.
order_counter = Counter(flower_orders)
order_counter


### 2. Count how many objects have color W in them.
count_with_w = sum('W' in order for order in flower_orders)
count_with_w

### 3. Make histogram of colors

# Hint from JohnP - Itertools has a permutation function that might help with these next two.
### 4. Rank the pairs of colors in each order regardless of how many colors are in an order.
### 5. Rank the triplets of colors in each order regardless of how many colors are in an order.
### 6. Make a dictionary with key=”color” and values = “what other colors it is ordered with”.
### 7. Make a graph showing the probability of having an edge between two colors based on how often they co-occur.  (a numpy square matrix)
### 8. Make 10 business questions related to the questions we asked above.

