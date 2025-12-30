import argparse

parser = argparse.ArgumentParser(description="Example script")
parser.add_argument("--name", default="Richard",help="Your name")
parser.add_argument("--age", type=int, default=20, help="Your age")

args = parser.parse_args()

print(f"Name: {args.name}")
print(f"Age: {args.age}")


# alternative
#import sys
#name = sys.argv[1]
#age = sys.argv[2]
#print(name, age)
