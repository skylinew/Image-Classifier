


'''

            ---------- ALL FUNCTION SIGNATURES MUST BE THE SAME --------------

                - A list of features functions

                - Perceptron algorithm stores these functions in a list
                  and executes each one via loop

                - typeflag: 0 for digits, 1 for faces

            ------------------------------------------------------------------

'''




'''
Counts total number of # (hash tag) symbols
'''
def hashtag_count(datum, typeflag):
    if typeflag == 0:
        width = height = 28
    else:
        width = 60
        height = 74
    count = 0
    for i in range(height):
        for j in range(width):
            if datum.pixels[j][i] == 2:
                count += 1

    return count

'''
    !!! FOR USE WITH DIGITS ONLY !!!
    
Counts total number of + (non-edge) symbols
'''
def plus_count(datum, typeflag):
    if typeflag == 0:
        width = height = 28
    else:
        width = 60
        height = 74
    count = 0
    for i in range(height):
        for j in range(width):
            if datum.pixels[j][i] == 1:
                count += 1

    return count


'''
Counts total number of ' ' (blank) symbols
'''
def space_count(datum, typeflag):
    if typeflag == 0:
        width = height = 28
    else:
        width = 60
        height = 74
    count = 0
    for i in range(height):
        for j in range(width):
            if datum.pixels[j][i] == 0:
                count += 1
    return count


'''
Default ratio, ratio of blank to non-blank symbols
'''
def ratio(datum, typeflag):
    if typeflag == 0:
        width = height = 28
    else:
        width = 60
        height = 74

    blank_count = 0
    non_blank_count = 0

    for i in range(height):
        for j in range(width):
            if datum.pixels[j][i] == 1 or datum.pixels[j][i] == 2:
                non_blank_count += 1
            else:
                blank_count += 1

    return blank_count / non_blank_count


'''
    !!! FOR USE WITH DIGITS ONLY !!!
                
Ratio of '+' to '#' symbols
'''
def edge_ratio(datum, typeflag):
    if typeflag == 0:
        width = height = 28
    else:
        width = 60
        height = 74

    blank_count = 0
    non_blank_count = 0

    for i in range(height):
        for j in range(width):
            if datum.pixels[j][i] == 1 or datum.pixels[j][i] == 2:
                non_blank_count += 1
            else:
                blank_count += 1

    return blank_count / non_blank_count




'''

               Digit Functions



'''





'''
        For digit '1'

        Finds the variance in non-blank horizontal line lengths
'''
# For 1's, if the average horizontal length of consecutive non-blank symbols is <= 4, return 1, else return -1
def avg_horizontal_line_length(datum, typeflag):
    if typeflag == 0:
        width = height = 28
    else:
        width = 60
        height = 74

    # a list containing the lengths of all horizontal lines in the image
    # horizontal lines are just consecutive non-blank symbols in a row
    horizontal_lines = []
    for i in range(height):
        count = 0
        for j in range(width):
            if datum.pixels[i][j] != 0:
                count += 1
        horizontal_lines.append(count)

    sum = 0
    for line in horizontal_lines:
        sum += line
    avg = sum / len(horizontal_lines)

    return avg


'''
        For digit '1'

        Finds the variance in non-blank horizontal line lengths
'''
def variance_horizontal_line_length(datum, typeflag):
    if typeflag == 0:
        width = height = 28
    else:
        width = 60
        height = 74

    # a list containing the lengths of all horizontal lines in the image
    # horizontal lines are just consecutive non-blank symbols in a row
    horizontal_lines = []
    for i in range(height):
        count = 0
        for j in range(width):
            if datum.pixels[i][j] != 0:
                count += 1
        horizontal_lines.append(count)

    sum = 0
    for line in horizontal_lines:
        sum += line
    avg = sum / len(horizontal_lines)

    squared_diffs = 0

    for line in horizontal_lines:
        squared_diffs += ((avg - line) * (avg - line))

    return squared_diffs/len(horizontal_lines)
