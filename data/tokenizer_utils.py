from typing import List, Dict

class Solution:
    def tokenize_numbers(self, m: List[int], vocab: Dict[str, int]) -> List[List[str]]:
        # Tokenize each number using greedy left-to-right longest match.
        # Return a list of token lists showing how each number gets split.
        ans =[]
        for num in m:
            res = self.tokenize(str(num),vocab)
            ans.append(res)
        return ans

    def tokenize(self,numstr,vocab):
        res = []
        l ,r= 0,1
        n = len(numstr)
        
        while l<n:
            best = None
            for length in range(n-l,0,-1):
                substr = numstr[l:l+length]
                if substr in vocab:
                    best = substr
                    break
            if best is None:
                res.append(numstr[l])
                l+=1
            else:
                res.append(best)
                l+=len(best)
                    
    
        return res
                

    def count_tokens(self, text: str, vocab: Dict[str, int]) -> int:
        # Count how many tokens the text uses with greedy tokenization.
        # Use greedy left-to-right longest match.
        return len(self.tokenize(text,vocab))

    def fertility_score(self, text: str, vocab: Dict[str, int]) -> float:
        # Compute tokens-per-word ratio (fertility).
        # Higher = more expensive and less efficient.
        # Round to 4 decimal places.
        tokens = self.tokenize(text,vocab)
        words= text.split()
        return round(len(tokens)/len(words),4)
