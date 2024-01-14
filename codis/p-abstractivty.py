"""
    Function that returns the length of ALL the longest common subsequence of two strings
    @param X the first string
    @param Y the second string
    @return the length of the longest common subsequence of X and Y
"""
def LCS(X , Y):
    m, n = len(X), len(Y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_length = dp[m][n]

    if lcs_length == 0:
        return 0, 0, []

    # Find all the LCS subsequences
    subsequences = []
    stack = [(m, n, [])]
    while stack:
        i, j, seq = stack.pop()
        if i == 0 or j == 0:
            subsequences.append(seq[::-1])
        elif X[i - 1] == Y[j - 1]:
            new_seq = seq.copy()
            new_seq.append(X[i - 1])
            stack.append((i - 1, j - 1, new_seq))
        else:
            if dp[i - 1][j] >= dp[i][j - 1]:
                stack.append((i - 1, j, seq))
            if dp[i][j - 1] >= dp[i - 1][j]:
                stack.append((i, j - 1, seq))

    # Count the number of LCS subsequences
    num_subsequences = len(subsequences)

    return lcs_length, num_subsequences
    

"""
    Function that returns the LCS segments common of given original tex and summary
    @param T the original text
    @param S the summary
    @return a list of LCS length segments
"""
def F (T, S):
    lcs_length, num_subsequences = LCS(T, S)
    return lcs_length, num_subsequences



"""
    function to calculate the p-abstractivity of a summary S of a text T
    @param T the original text
    @param S the summary
        @param p the abastractivity parameter: p > 1 penalizes longer extractive sequences and reduce importance of the shorter extractive sequences
    @return  a real number in [0,1]
"""
def abstractivity (T, S, p):
    lcs_length, num_subsequences = F(T, S)
    denom = len(S) ** p
    if denom == 0:
        return 0
    else:
        return 1 - ((lcs_length ** p)  * num_subsequences/ denom)