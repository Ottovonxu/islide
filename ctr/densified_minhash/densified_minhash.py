from sklearn.utils import murmurhash3_32 as mmh
import numpy as np
import pdb
from sklearn.utils import murmurhash3_32

def Hfunction(m, seed=None):
    #if seed is None:
    #    seed = np.random.randint(0,100000)
    return lambda x : murmurhash3_32(key=x, seed=seed, positive=True) % m


class Densified_MinHash():
    def __init__(self, K, D, bbit=8, seed=0, num_seed=0, hashFull=False):
        self.K = K
        self.D = D # this is the target dimension. not the data dimension
        #self.bbit = bbit
        self.seed = seed
        self.bucket_size = int(self.D / self.K)
        self.hashFull = hashFull
        

        self.big_prime = 100001893
        self.random_number_generator = Hfunction(self.big_prime, num_seed)
        #Generating random numbers for rotation densified embedding
        #self.kidx_mul, self.attempt_mul, self.value_mul, self.bias_mul = [self.random_number_generator(i) for i in range(1,5)]
        #self.axis0 = np.arange(0, self.K, dtype=np.int32)
        #np.random.shuffle(self.axis0)
        
        # vectorized independent min hashes
        # we will use 4-independent hash functions for more randomness
        indices = range(0,self.K*4)
        self.A1s = np.array([self.random_number_generator(i+5) for i in indices[0:self.K]]).reshape(1, self.K)
        self.A2s = np.array([self.random_number_generator(i*10) for i in indices[self.K:2*self.K]]).reshape(1, self.K)  
        self.A3s = np.array([self.random_number_generator(i*100) for i in indices[2*self.K:3*self.K]]).reshape(1, self.K)  
        self.Bs = np.array([self.random_number_generator(i*178+5) for i in indices[3*self.K:]]).reshape(1, self.K)  
        #print("DMH:", "seed:", seed, "num_seed", num_seed, "hashFull:", self.hashFull)
        #print("Config of minhash functions:")
        #print(self.A1s)
        #print(self.A2s)
        #print(self.A3s)
        #print(self.Bs)
        #print(self.axis0)
    def hash_func(self, seed):
        return lambda x: mmh(key=x, seed=seed, positive=True) % self.D

    def hash_bin_to_bin(self, bin_id, attempt, seed):
        key = str(attempt) + "." + str(bin_id)
        return mmh(key=key, seed=seed, positive=True) % self.K

    def convert_to_bit_array(self, k_hashes):
        k_hashes = [i for i in k_hashes if i != -1]
        one_hot = np.zeros(self.D)
        for k in k_hashes:
            one_hot[k] =1
        return one_hot

    def get_hashed(self, word_set):
        k_hashes = [-1 for _ in range(self.K)]
        hash_func = self.hash_func(self.seed)
        for w in word_set:
            hash_val = hash_func(w)
            idx = int(1. * hash_val * self.K / self.D)
            if hash_val > k_hashes[idx]:
                k_hashes[idx] = hash_val
        # optimal densified hashing for empty bins
        for idx in range(self.K):
            if k_hashes[idx] == -1:
                attempt = 1
                new_bin = self.hash_bin_to_bin(idx, attempt, self.seed)
                while k_hashes[new_bin] == -1:
                    attempt += 1
                    new_bin = self.hash_bin_to_bin(idx, attempt, self.seed)
                k_hashes[idx] = ((idx + 4 * attempt) + (k_hashes[new_bin] + 3 * attempt)) % int(self.D)
                #if self.hashFull:
                #    k_hashes[idx] =  mmh(key=str(idx)+"."+str(k_hashes[new_bin])+"."+str(attempt), seed=self.seed, positive=True) % self.D
                #else:
                #    k_hashes[idx] =   mmh(key=str(idx)+"."+str(k_hashes[new_bin])+"."+str(attempt), seed=self.seed, positive=True) \
                #                    % self.bucket_size + idx * self.bucket_size

        return k_hashes

    #def get_hashed_faster(self, word_set):
    #    ''' this gives one permutation hash densified by rotation
    #        has more variance and it also reflects in the accuracy
    #    '''
    #    hf = self.hash_func(self.seed)
    #    original_indices = [hf(w) for w in word_set] # using for loop can it be saved?
    #    original_vector = np.zeros(self.D)
    #    original_vector[original_indices] = 1
    #    matrix = np.reshape(original_vector, (self.K, self.bucket_size))
    #    shuffled = matrix[self.axis0]
    #    recover = np.zeros(self.K, dtype=np.int32)
    #    recover[self.axis0] = np.arange(0,self.K)
    #    neworder = np.argmax(shuffled, axis=1)
    #    mask = (neworder == 0)
    #    idx = np.where(~mask,np.arange(0,len(neworder)),0)
    #    fillidx = np.maximum.accumulate(idx,axis=0, out=idx)
    #    newmask = np.multiply.accumulate(mask)
    #    fillidx = fillidx + newmask * fillidx[-1]
    #    attempt = np.arange(0,len(fillidx)) - fillidx
    #    attempt = attempt % len(fillidx)
    #    changes = (fillidx * self.kidx_mul + attempt * self.attempt_mul + neworder[fillidx] * self.value_mul + self.bias_mul)% self.big_prime % self.bucket_size
    #    finalhash = changes * mask + neworder
    #    shuffled[np.arange(0,self.K, dtype=np.int32), finalhash] = 1
    #    recovered_matrix = shuffled[recover]
    #    embedding = recovered_matrix.reshape(self.D,)
    #    return embedding
        
    def get_hashed_4universal(self, word_set):
        '''  
          simply compute independent min hash using 4-universal hash function and hash it to the range
       '''
        if type(word_set[0]) == int:
            original_indices = word_set
        else:
            hf = self.hash_func(self.seed)
            original_indices = [hf(w) for w in word_set] # using for loop can it be saved? This is done for pairwise compatibility.
        ws = np.array(original_indices).reshape(len(word_set), 1)
        hashes = (np.matmul(ws, self.A1s) + np.matmul(np.square(ws), self.A2s) + np.matmul(np.power(ws, 3), self.A3s) + self.Bs)% self.big_prime
        min_hashes = np.min(hashes, axis=0)
        assert(len(min_hashes) == self.K)
        
        if self.hashFull:
          min_hashes = min_hashes % self.D
        else:
          # hash to buckets
          min_hashes = min_hashes % self.bucket_size + np.arange(0, self.K) * self.bucket_size
        embedding = np.zeros(self.D)
        embedding[min_hashes] =1
        return embedding
        
        



if __name__ == '__main__':
    s1 = [480,923,106]
    s2 = [480,106,373]
    D = 1000
    HD = 100
    K = 10
    print("Jaccard", len([a for a in s1 if a in s2])/ (len(s1) + len(s2) - len([a for a in s1 if a in s2])))

    vs = []
    for i in range(0,1000):
        #DMH = Densified_MinHash(K, HD, seed=i)
        DMH = Densified_MinHash(K, HD, seed=mmh(i,30,positive=True), num_seed=mmh(i,20,positive=True), hashFull=True)
        #x1 = DMH.get_hashed(s1)
        #x2 = DMH.get_hashed(s2)
        #xs1 = DMH.convert_to_bit_array(x1)
        #xs2 = DMH.convert_to_bit_array(x2)
        #xs1 = DMH.get_hashed_faster(s1)
        #xs2 = DMH.get_hashed_faster(s2)
        xs1 = DMH.get_hashed_4universal(s1)
        xs2 = DMH.get_hashed_4universal(s2)
        vs.append(np.dot(xs1,xs2)/K)
    print(np.mean(vs), np.std(vs))
