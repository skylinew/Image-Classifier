


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


