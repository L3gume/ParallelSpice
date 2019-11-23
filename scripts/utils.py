def zeros(rows, cols):
    return [[0.0 for x in range(cols)] for y in range(rows)]

def generateMesh(N, M):
    
    if(N < 2 or M < 2 or N%1 != 0 or M%1 != 0):
        raise Exception("Invalid dims")
    
    tNodes = N*M
    tBranches = M*(N-1) + N*(M-1)
    
    A = zeros(tNodes, tBranches)
    
    # populate incidence matrix
    for node in range(tNodes):
        for current in range(tBranches):
            
            # populate horizontal positive flowing currents
            if current < N*(M-1):                             # validate horizonal current
                
                # positive flow
                if node%M != M-1:                             # skip horizontal boundary
                    if node%M == current%(M-1):               # verify current and node position match
                        if node//M == current//(M-1):         # validate current and node rank match
                            A[node][current] = 1.0
                
                # negative flow
                if node%M != 0:                               # skip horizontal boundary
                    if (node-1)%M == (current)%(M-1):         # verify current and node position match
                        if (node-1)//M == (current)//(M-1):   # validate current and node rank match
                            A[node][current] = -1.0

            # populate vertical flowing currents
            else:
                
                #positive flow
                if current % (N*(M-1)) == node:
                    A[node][current] = 1.0
                    
                # negative flow
                if node < tNodes - M:
                    if current % (N*(M-1)) == node:
                        A[node+M][current] = -1.0
            
    return A

def applyVoltages(M):
    
    # RHS stimulation
    # add additional current to each row
    rows, cols = len(M), len(M[0])
    A = zeros(rows, cols+1)

    # copy main mesh
    for row in range(rows):
        for col in range(cols):
            A[row][col] = M[row][col]

    # implant stimulated circuit   
    A[0][-1] = -1.0
    A[-1][-1] = 1.0
    

    #LHS stim
    rows, cols = len(A), len(A[0])
    C = zeros(rows, cols+1)

    for row in range(rows):
        for col in range(cols):
            C[row][col+1] = A[row][col]
    
    C[0][0] = -1.0
    C[-1][-1] = 1.0        
    
    # ground
    C = C[:-1]
        
    return C


