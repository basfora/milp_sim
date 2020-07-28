from milp_risk.scripts.danger import MyDanger
import milp_mespp.core.extract_info as ext
import sys
print(sys.path)

# parameters
deadline = 10
g = ext.get_graph_04()
what = 'fire'
plot_for_me = True

# init
# hazard = MyHazard(g, deadline, what)
# hazard.simulate(3)
danger = MyDanger(g, deadline, plot_for_me)