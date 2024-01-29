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