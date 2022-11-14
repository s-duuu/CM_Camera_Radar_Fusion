from dataclasses import dataclass

@dataclass
class object:
    index:int = None
    distance:int = None


if __name__ == '__main__':
    object_list = []
    for i in range(5):
        var = object()
        var.index = 5 - i
        var.distance = i
        object_list.append(var)
    
    print(min(object_list))