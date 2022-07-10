import random



if __name__ == "__main__":
    pairs = []
    result = []
    for i in range(100):
        temp = []
        temp.append(random.randint(0, 1))
        temp.append(random.randint(0, 1))
        pairs.append(temp)
        result.append(temp[0] * temp[1])
    
    with open('in.txt', 'w') as file:
        for i in pairs:
            file.write(f'{i[0]} {i[1]}\n')
    
    with open('out.txt', 'w') as file:
        for i in result:
            file.write(f'{i}\n')
