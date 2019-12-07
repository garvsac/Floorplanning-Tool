# HeKant
## Contraction:Done
    a) initialize the good vertices
    b) finding the contractible neighbour
    c) storing the y and z vertices
    d) storing the neighbours of contracted vertices
    e) updating the adjacency matrix
    f) updating the good vertices
    Data for contraction: u:contracted,v:neighbor, y:upper neighbor, z:bottom neighbor, u neighbors.
## Expansion:Done
    a) Algo for neighbors in order
    b) Function to find clockwise neighbor
    c) Match expansion patterns
    d) Update matrix
## Other Stuff:15-Dec
    a) Display NESW in correct positions 
    b) tkinter + matplotlib
    c) Triangulation
    d) REL -> RDG: Done
    e) Pseudo Code
    f) Validation/Check for REL
    
## Known Issues:
    a) Bug in Cases. Not Working.
        0 1
        1 2
        0 3
        3 4
        4 2
        5 6
        6 4
        6 7
        7 8
        5 9
        9 8
        3 1
        4 1
        5 0
        6 0
        6 3
        7 4
        8 4
        8 2
        9 6
        9 7
    b) Planarity Issue after Contraction
