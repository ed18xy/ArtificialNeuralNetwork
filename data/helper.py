with open('data5.out', 'r') as f1, open('data6.out', 'r') as f2, open('data13.out', 'w') as f3:
    for line1, line2 in zip(f1, f2):
        f3.write(line1.strip() + " " + line2.strip() + "\n")
