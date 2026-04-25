from typing import List
from collections import defaultdict

class Solution:
    def get_merges(self, corpus: str, num_merges: int) -> List[List[str]]:
        # 1. Split corpus into a list of individual characters
        # 2. For each merge step:
        #    a. Count frequency of all adjacent token pairs
        #    b. Find the most frequent pair (break ties lexicographically)
        #    c. Merge all non-overlapping occurrences left to right
        #    d. Record the merge as [token_a, token_b]
        # 3. Return the list of merges performed
        tokens = list(corpus)
        # print(tokens)
        merges = []
        for _ in range(num_merges):
            # print(tokens,'tok')
            if len(tokens)<2:
                break
            pairs = defaultdict(int)
            
            for i in range(len(tokens)-1):
                pairs[(tokens[i],tokens[i+1])]+=1
           
            if not pairs:
                break
            maxf = max(pairs.values())
            best_pairs = [pair for pair in pairs if pairs[pair]==maxf ]
            best_pairs.sort()
            # print(best_pairs)
            best_pair = best_pairs[0]
            merges.append([best_pair[0],best_pair[1]])
            ntk = []
            i =0 
            while i <len(tokens):
                if i<len(tokens) -1 and tokens[i]==best_pair[0] and tokens[i+1]==best_pair[1]:
                    i+=2
                    ntk.append(best_pair[0]+best_pair[1])
                else:
                    ntk.append(tokens[i])
                    i+=1
            # print(ntk,'ntk')
            tokens = ntk



            


            
        return merges
