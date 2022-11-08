import cProfile
import pstats
import io


pr = cProfile.Profile()
pr.enable()

exec(open("working.py").read())

pr.disable()
s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
ps.print_stats()

with open('profile_results.txt', 'w+') as f:
    f.write(s.getvalue())