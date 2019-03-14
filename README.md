# Min-cost flow

This is the code that [this](https://blog.farnasirim.ir/2019/03/python-min-cost-max-flow-and-loose-end.html)
blog post talks about.

## Usage


```bash
python3 ./mcf.py 2 3 small.csv matching-output.csv info-output.txt
python3 ./mcf.py 2 3 large.csv matching-output.csv info-output.txt
```

`2` and `3` are parameters related to the input. You won't need to 
change them if you intend to use the supplied data.

`matching-output.csv` and `info-ouptut.txt` are two files that the program
writes upon exit. If you change something in `./mcf.py`, you would probably
like the output files to stay consistent.

## License
MIT