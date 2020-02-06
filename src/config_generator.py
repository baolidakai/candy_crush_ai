"""Generator for config file."""
import utils
import random
import sys

def main():
  print('Example usage: python config_generator.py 200 10 5 train/config1.txt will write 100 rows, 10 digits per row, 5 colors to ../config/train/config1.txt')
  assert len(sys.argv) == 5
  m = int(sys.argv[1])
  n = int(sys.argv[2])
  num_colors = int(sys.argv[3])
  assert 1 < num_colors <= utils.MAX_COLOR
  dst_file = '../config/' + sys.argv[4]
  print('Writing %d rows with %d digits per row and %d colors to %s' % (m, n, num_colors, dst_file))
  with open(dst_file, 'w') as fout:
    for i in range(m):
      for j in range(n):
        color = random.randint(0, num_colors - 1)
        fout.write(str(color))
      fout.write('\n')

if __name__ == '__main__':
  main()

