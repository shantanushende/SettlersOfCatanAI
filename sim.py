
import copy
import random
import numpy as np
import pprint
from collections import defaultdict
import time

pp = pprint.PrettyPrinter(indent=4)

class Cube(object):

    def __init__(self, matrix=None,code=None):
        if matrix == None:
            self.matrix = code
        if code == None:
            self.matrix = ','.join([str(i) for i in matrix])
        

    def show(self):
        return self.matrix



class Hexgrid(object):

    def __init__(self, depth):
        
        self.depth = depth
        self.cubes = list()
        self.create()
        

    def create(self):
        n = self.depth
        self.cubes += [Cube([x,y,z]) for x in range(-n,n+1) for y in range(-n,n+1) for z in range(-n,n+1) if sum([x,y,z]) == 0]
        print( 'created {} cubes'.format( len(self.cubes)))

    def show(self):
        return [c.show() for c in self.cubes]

## utility function
def add_vec(a,b):
    # a: list
    # b: list
    if type(a) == list and type(b) == list:
        assert(len(a)==len(b))
        return list(sum(z) for z in zip(a,b))
    # a: list
    # b: string
    if type(a) == list and type(b) == str:
        arr_b = [int(i) for i in b.split(',')]
        assert(len(a)==len(arr_b))
        return list(sum(z) for z in zip(a,arr_b))
    raise Exception('add_vec params should be of type list,list or list,string')
        
    

#utility function
def random_order_array(source):
    arr = []
    while source:
        r = random.randrange(len(source))
        arr.append(source.pop(r))
    return arr

#utility , note these are intentionally ordered clockwise 
CDIR = [
            [+1, 0, -1], [+1, -1, 0], [0, -1, +1], 
            [-1, 0, +1], [-1, +1, 0], [0, +1, -1],   
        ]

RMAP = {
    'Wheat':[1,0,0,0,0],
    'Brick':[0,1,0,0,0],
    'Rock' :[0,0,1,0,0],
    'Wood' :[0,0,0,1,0],
    'Sheep':[0,0,0,0,1],
}

PRICES = {
    'city': [-2,0,-3,0,0],
    'sett': [-1,-1,0,-1,-1],
    'road': [0,-1,0,-1,0],
}

DOTS = {
    '0':0,
    '2':1,
    '3':2,
    '4':3,
    '5':4,
    '6':5,
    '8':5,
    '9':4,
    '10':3,
    '11':2,
    '12':1,
}

TILE_DOT_VALUES = [5,2,6,3,8,10,9,12,11,4,8,10,9,4,5,6,3,11]


DEV_CARDS=random.sample((['knight'] * 14 + \
                            ['monopoly'] * 2 + \
                            ['year_plenty'] * 2 + \
                            ['road_building'] * 2 + \
                            ['victory_point'] * 5),25)


def roll_dice():
    return random.randrange(1,7) + random.randrange(1,7)

 
class Board(object):               
    def __init__(self,start_cube,hex_grid):
        self.layout = dict()
        self.depth = hex_grid.depth
        self.hex_grid_cubes = copy.deepcopy(hex_grid.cubes)
        self.start_cube = start_cube
        self.lay_out_board()

    def assign_resources(self):
        print('assign_resources!')

        r = [['Wheat'] * 4] + \
            [['Brick'] * 3] + \
            [['Rock'] * 3] + \
            [['Wood'] * 4] + \
            [['Sheep'] * 4] + \
            [['Desert'] * 1] 

        r = [val for sublist in r for val in sublist]
        return random_order_array(r)

    def clockwise_placement(self):
        sc = self.start_cube
        #return array of tiles in placement order beginning at input start_cube
        depth_val = copy.deepcopy(self.depth) + 1
        co = []
        #verify start cube sits at outer level of game
        assert( sum([abs(int(i)) for i in sc.split(',')]) == self.depth * 2)
        cubes = copy.deepcopy([c.matrix for c in self.hex_grid_cubes])
        this_cube = sc
        while cubes:
            co.append(this_cube)
            cubes.remove(this_cube)
            #select cube neighbor in clockwise position
            tfs = [all(abs(i) < depth_val for i in add_vec(cd,this_cube)) for cd in CDIR]     
            for i in range(6):
                if tfs[i] == False and tfs[(i+1)%6] == True:
                    d = (i+1)%6
            ncc = ','.join([str(i) for i in add_vec( CDIR[d], this_cube)])
            #when a circuit completes, move one step closer to center & resume clockwise motion
            if ncc not in cubes:            
                depth_val -= 1
                new_d = (d + 1 ) % 6
                ncc = ','.join([str(i) for i in add_vec( CDIR[new_d], this_cube)])
            this_cube = ncc
        return co

    def lay_out_board(self):
        co = self.clockwise_placement()
        #map together tiles, points, and resource
        resources = self.assign_resources()
        d_idx = resources.index('Desert')
        points = TILE_DOT_VALUES
        points.insert(d_idx,0)
        assert( len(resources) == len(points) == len(co) )
        self.layout = { co[i] : {'sc': str(points[i]) ,'r':resources[i]} for i in range(len(co)) }

    def show_board_layout(self):
        print( 'Board Layout: \n')
        pp.pprint(self.layout)


class Player(object):
    def __init__(self,label):
        self.label = label
        self.abbv = label[0]

    def show(self):
        print( 'Player {} '.format(self.label))


class State(object):

    def __init__(self, hex_grid=None, players=None, board=None,gs=None, this_player=None, parent=None, played_dc=False):
        if played_dc:
            self.played_dc = True
        else:
            self.played_dc = False
        self.players = ['A','B']
        if gs == None:
            self.hex_grid = copy.deepcopy(hex_grid)
            self.depth = hex_grid.depth
            self.graph = dict()
            self.board = board
            
            self.make_graph()
            self.gs = dict()
            self.make_game_state()
        else:
            self.gs = gs
            self.players = [p for p in self.gs['p']]
            
            self.children = []
        if this_player:
            self.turn = self.players.index(this_player)
        else:
            pass
        self._number_of_visits = 1.
        self._results = {p:0 for p in self.players}
        #self._untried_actions = None
        self.parent = parent


    def neighbors(self,cube):
        return list( add_vec(cd,cube.matrix) for cd in CDIR) 

    def cubes_adjacent_to_node(self,c,n):
        dirs = {
        '12': [[0,+1,-1],[+1,0,-1]],
        '2':  [[+1,0,-1],[+1,-1,0]],
        '4':  [[+1,-1,0],[0,-1,+1]],
        '6':  [[0,-1,+1],[-1,0,+1]],
        '8':  [[-1,0,+1],[-1,+1,0]],
        '10': [[-1,+1,0],[0,+1,-1]]
        }
        r = []

        for adj_node_mask in dirs[n]:
            adj_n = add_vec(adj_node_mask,c.matrix)
            #only return adjacent nodes on the board i.e. -2 <= {x,y,z} <= 2 
            if all(-3 < i < 3 for i in adj_n):
                r.append(Cube(adj_n))
        #return nodes on board, or adjacent to boundary
        return r

    def find_matches(self,gg,g):
        matches = []
        #dont calculate matches for edge pieces touching one cube
        if len(gg['c']) == 1:
            return matches
        for item in g:
            matrix_list = [ii.matrix for ii in item['c']]
            if set(matrix_list)==  set([c.matrix for c in gg['c']]):
                #dont return same node in match list
                if gg['n'] != item['n']:
                    matches.append(item)
        return matches

    def deduplicate_nodes(self,g):
        deep_copy_g = copy.deepcopy(g)
        ug = []
        for gg in deep_copy_g:
            #for each node find equivalents mapping to same cubes
            matches = self.find_matches(gg,ug)
            if matches:
                matches[0]['n'] = matches[0]['n'] + ';'+ gg['n']
            else:
                ug.append(gg)
        assert(  self.test_node_map(ug) )
        return ug

    def gather_nodes(self):
        g = []
        #loop through each cube
        for c in self.hex_grid.cubes:
            #loop through each node vertex
            for n in ['12','2','4','6','8','10']:
                #set up the graph node
                label = c.matrix + ',' + n
                n_dict = {'n':label}
                #get the cubes for node
                n_dict['c'] = [c]
                n_dict['c']+=self.cubes_adjacent_to_node(c,n)
                #add node to graph
                g.append(n_dict)
        return g

    def calc_edges(self, graph):

        g = {}
        #make dictionary
        for gg in graph:
            g[gg['n']] = {}
        
        assert( len(g) == 54)
        
        adj_map = {
        '12': ['10','2'],
        '2':  ['12','4'],
        '4':  ['2','6'],
        '6':  ['4','8'],
        '8':  ['6','10'],
        '10': ['8','12']
        }

        ref_nodes = [ k for k in copy.deepcopy(g) ]

        ## asign settlement/city position

        for n in g:
            g[n]['c'] = None
            g[n]['s'] = None
            g[n]['e'] = dict()

        for n in g:
            adjacents = []
            node_identifiers = n.split(';')

            #cycle through each cube in node signature
            for nid in node_identifiers:
                this_hex = copy.deepcopy(','.join(nid.split(',')[:3]))
                adj_hex_ordinals = adj_map[nid.split(',')[3]]

                #cycle through each adjacent ordinal
                for hex_ord in adj_hex_ordinals:
                    adj_hex_node = this_hex + ',' + hex_ord

                    #lookup full node encoding adjacents, conditionally add to adjacents
                    full_encoding = [ i for i in ref_nodes if adj_hex_node in i ]
                    assert( len(full_encoding) == 1 )
                    fe = full_encoding[0]
                    if fe not in adjacents:
                        adjacents.append(fe)

            #g[n]['e'] = [tuple([adj,None]) for adj in adjacents]
            for adj in adjacents:
                g[n]['e'][adj] = None
        return g
        
    def map_nodes_to_ports(self,graph):

        for n in graph:
            graph[n]['p'] = None

        ports = {
            'p1': {'loc': ['-2,1,1,6','-2,1,1,8'], 'val': None, 'res':None }, #randomly assigned
            'p2': {'loc': ['-2,2,0,8','-2,2,0,10'], 'val': None, 'res':None },#randomly assigned
            'p3': {'loc': ['-1,2,-1,10','-1,2,-1,12'], 'val': 2, 'res': 'Brick' }, #brick 2:1
            'p4': {'loc': ['1,1,-2,10','1,1,-2,12'], 'val': 2, 'res': 'Wood' },#wood 2:1
            'p5': {'loc': ['2,0,-2,12','2,0,-2,2'], 'val': None, 'res':None },#randomly assigned
            'p6': {'loc': ['2,-1,-1,2','2,-1,-1,4'], 'val': 2, 'res': 'Wheat' },#wheat 2:1
            'p7': {'loc': ['1,-2,1,2','1,-2,1,4'], 'val': 2, 'res': 'Rock' },#rock 2:1
            'p8': {'loc': ['0,-2,2,4','0,-2,2,6'], 'val': None, 'res':None },#randomly assigned
            'p9': {'loc': ['-1,-1,2,6','-1,-1,2,8'], 'val': 2, 'res': 'Sheep' },#sheep 2:1
        }

        assignable_ports = [
            {'val': 3, 'res':None },
            {'val': 3, 'res':None },
            {'val': 3, 'res':None },
            {'val': 3, 'res':None },
            {'val': 2, 'res':'Wheat' },
            {'val': 2, 'res':'Brick' },
            {'val': 2, 'res':'Rock' },
            {'val': 2, 'res':'Wood' },
            {'val': 2, 'res':'Sheep' },
        ]

        for p in ports:
            assert( ports[p]['loc'][0].split(',')[:3] == ports[p]['loc'][1].split(',')[:3] )
            node1 = [key for key, val in graph.items() if ports[p]['loc'][0] in key]
            node2 = [key for key, val in graph.items() if ports[p]['loc'][1] in key]
            assert( len(node1) == 1 and len(node2) == 1 )
            node1 = node1[0]
            node2 = node2[0]
            assert(node1 != node2)
            if ports[p]['val'] == None and ports[p]['res'] == None:
                ap = assignable_ports.pop(random.randrange(len(assignable_ports)))
                o = tuple([ap['val'],ap['res']])
            else:
                o = tuple([ports[p]['val'],ports[p]['res']]) 
            graph[node1]['p'] = o
        assert(len([n for n in graph if graph[n]['p'] != None]) == 9)
        return graph

    def random_assign(self, target, source):
        assert(len(source)>=len(target))
        mapping = { i : None for i in target }
        for val in mapping:
            r = random.randrange(len(source))
            mapping[val] = source.pop(r)
        return mapping

    def test_node_map(self, ug):
        if len(ug) != 54:
            raise Exception('problem! ug != 54')
        for gg in ug:
            if len( gg['n'].split(';')) != len(gg['c']):
                raise Exception('problem!')
        return True

    def make_graph(self): 
        #gather nodes
        g = self.gather_nodes()
        #depuplicate nodes
        ug = self.deduplicate_nodes(g)
        #add edges to make graph
        graph = self.calc_edges(ug)

        graph = self.map_nodes_to_ports(graph)

        self.graph = graph

    def show_graph(self):
        print('Show Graph {} nodes \n'.format(len(self.graph)))
        pp.pprint(self.graph)

    def make_game_state(self):

        g = copy.deepcopy(self.graph)
        b = copy.deepcopy(self.board)
        gs = {'g':g,'b':b.layout}
        pd = {str(i):[0]*5 for i in range(1,13) }
        pd['hand'] = [0]*len(RMAP)  # Important: resource array order is [wheat, brick, rock, wood, sheep] see RMAP
        pd['v'] = 0
        pd['dev_hand'] = {i:0 for i in list(set(DEV_CARDS)) }
        pd['pieces'] = {'c':4,'s':5,'r':15}

        gs['p'] = dict()
        for p in self.players:
            gs['p'][p] = copy.deepcopy(pd)
        gs['turn'] = 0
        gs['dev_deck'] = DEV_CARDS
        gs['longest_road'] = tuple([5,None])
        self.gs = gs

    def show_game_state(self):
        print( 'showing game state \n' )
        pp.pprint(self.gs)

    def adjust_victory_points(self,gs,player,points):
        gs['p'][player]['v'] += points
        return gs

    def pay_for_purchase(self,gs,player,price):
        #print( 'before pay_for_purchase ', gs['p'][player]['hand'] )
        gs['p'][player]['hand'] = add_vec(copy.deepcopy(gs['p'][player]['hand']),price)
        #print( 'after pay_for_purchase ', gs['p'][player]['hand'] )

        return gs

    def place_settlement(self,player,n,gs, redeem_resource_cards=False):
        gs = copy.deepcopy(gs)
        gs['g'][n]['s'] = player
        gs = self.update_role_val_game_state(player,n,gs,redeem_resource_cards=redeem_resource_cards)
        gs = self.adjust_victory_points(gs, player,1)
        gs = self.pay_for_purchase(gs,player, PRICES['sett'])
        gs['p'][player]['pieces']['s'] -= 1
        return gs

    def place_city(self,player,n,gs):
        gs = copy.deepcopy(gs)
        assert(gs['g'][n]['s'] == player) 
        gs['g'][n]['c'] = player
        gs = self.update_role_val_game_state(player,n,gs)
        gs = self.adjust_victory_points(gs, player,2)
        gs = self.pay_for_purchase(gs,player, PRICES['city'])
        gs['p'][player]['pieces']['c'] -= 1
        gs['p'][player]['pieces']['s'] += 1
        return gs

    def place_road(self,player,n,dest,gs):
        gs = copy.deepcopy(gs)
        gs['g'][n]['e'][dest] = player
        gs['g'][dest]['e'][n] = player
        gs['p'][player]['pieces']['r'] -= 1
        assert( gs['g'][n]['e'][dest] == gs['g'][dest]['e'][n]  ) 
        return gs

    def update_role_val_game_state(self,player,n,gs,redeem_resource_cards=False):
        for t in n.split(';'):
            t = ','.join(t.split(',')[:3])
            card_type = gs['b'][t]['r']
            #update roll map for player 
            roll_val = gs['b'][t]['sc']         
            if roll_val != '0':
                gs['p'][player][roll_val] = add_vec(gs['p'][player][roll_val],RMAP[card_type])
            #add card to hand, only not true on first of 2 initial placements 
            if redeem_resource_cards:
                if card_type != 'Desert':
                    gs['p'][player]['hand'] = add_vec(gs['p'][player]['hand'],RMAP[card_type])
        return gs

    def adj_nodes_free(self,n,gs):
        # adjacent nodes do not have settlement or city  
        if any([ gs['g'][e]['s'] != None or gs['g'][e]['c'] != None for e in gs['g'][n]['e']]):
            return False
        return True

    def no_blocking_rds(self,n,gs):
        # 2 or more edges not occupied by opponent roads
        this_player = self.players[self.gs['turn']]
        if len([e for e in gs['g'][n]['e'] if gs['g'][n]['e'][e] != None  and gs['g'][n]['e'][e] != this_player]) >= 2:
            return False
        return True

    def has_road_access(self,n,player,gs):
        # player has a road to this node
        return any([e for e in gs['g'][n]['e'] if gs['g'][n]['e'][e] == player ])

    
    def collect_card(self,gs,player,card_type):
        gs['p'][player]['hand'] = add_vec(copy.deepcopy(gs['p'][player]['hand']),RMAP[card_type])
        return gs

    def make_bank_trade(self,gs,player,r,card_type):
        gs = copy.deepcopy(gs)
        price = [i * -4 for i in RMAP[r]]
        gs = self.pay_for_purchase(gs,player, price)
        gs = self.collect_card(gs,player,card_type)
        return gs

    def get_bank_trade_moves(self,gs,player):
        moves = []
        R_VAL = list(RMAP.keys())
        #print('R ', R_VAL)
        for i in range(len(gs['p'][player]['hand'])):
            if gs['p'][player]['hand'][i] >= 4:
                am = R_VAL[i] 
                moves.append(am)
        return moves

    def get_port_trade_moves(self,gs,player):
        #print('port_trade_moves!')
        port_trade_moves = []
        for n in gs['g']:
            if gs['g'][n]['s'] == player or gs['g'][n]['s'] == player:
                if gs['g'][n]['p'] != None:
                    port_trade_policy = gs['g'][n]['p']
                    #is resource port
                    if port_trade_policy[1] != None:
                        #print('player {} hand {} policy {}'.format( player, gs['p'][player]['hand'], port_trade_policy ))
                        #can afford resource port trade 
                        price = [ii * port_trade_policy[0] * -1 for ii in RMAP[port_trade_policy[1]]]

                        #print( 'port_trade_policy_v --> ', price )
                        if all([i >= 0 for i in add_vec(copy.deepcopy(gs['p'][player]['hand']),price)]):
                            port_trade_moves.append(port_trade_policy) 
                    else:
                        for resource in list(RMAP.keys()):
                            price = [ii * (-3) for ii in RMAP[resource]]
                            if all([i >= 0 for i in add_vec(copy.deepcopy(gs['p'][player]['hand']),price)]):
                                port_trade_moves.append(tuple([3,resource]))
                                

                    
        return port_trade_moves

    def make_port_trade(self,gs,player,port_trade_policy,card_type):
        gs = copy.deepcopy(gs)
        price = [ii * port_trade_policy[0] * -1 for ii in RMAP[port_trade_policy[1]]]
        #print( 'before pay_for_purchase ',  )
        gs = self.pay_for_purchase(gs,player, price)
        gs = self.collect_card(gs,player,card_type)
        return gs


    def longest_road_check(self,gs,player):
        
        pm_gs=copy.deepcopy(gs)
        if 15 - gs['p'][player]['pieces']['r'] < gs['longest_road'][0]:
            return pm_gs
        tlr = 0
        for n in gs['g']:
            #player has road in this location
            if player in [gs['g'][n]['e'][e] for e in gs['g'][n]['e'] if gs['g'][n]['e'][e] == player]:
                lr = 0
                ol = [tuple([n,0])]
                cl = []
                while ol:
                    c_n = ol.pop()
                    cl.append(c_n[0])
                    if c_n[1] > lr:
                        lr = c_n[1]
                    for ee in gs['g'][c_n[0]]['e']:
                        #get onward roads from here for player
                        if gs['g'][c_n[0]]['e'][ee] == player:
                            #explore node if not visited
                            if ee not in cl:
                                ol.append(tuple([ee,c_n[1]+1]))
                #update lr if warranted
                if lr > tlr:
                    tlr = lr



        if tlr > gs['longest_road'][0]:       
            if pm_gs['longest_road'][1] == player:
                pm_gs['longest_road'] =  tuple([tlr,player])
            elif pm_gs['longest_road'][1] == None:
                pm_gs['longest_road'] = tuple([tlr,player])
                pm_gs = self.adjust_victory_points(pm_gs, player,2)


            else:
                if player == 'A':
                    other_player = 'B'
                else:
                    other_player = 'A'
                pm_gs['longest_road'] = tuple([tlr,player])
                pm_gs = self.adjust_victory_points(pm_gs, player,2)
                pm_gs = self.adjust_victory_points(pm_gs, other_player,-2)
             
        return pm_gs


    def get_available_moves(self):        
        gs = copy.deepcopy(self.gs)
        player = self.players[self.gs['turn']]


        assert( all([i >= 0 for i in gs['p'][player]['hand'] ]) )

        #return
        am_city = []
        am_sett = []
        am_road = []

        ##can afford city
        if gs['p'][player]['pieces']['c'] >= 1:
            if all([i >= 0 for i in add_vec(copy.deepcopy(gs['p'][player]['hand']),[-2,0,-3,0,0])]):
                for n in gs['g']:
                    #can build city if there is a settlement
                    if gs['g'][n]['s'] == player:
                        am_city.append(n)
        
        ##can afford settlement
        if gs['p'][player]['pieces']['s'] >= 1:
            if all([i >= 0 for i in add_vec(copy.deepcopy(gs['p'][player]['hand']),[-1,-1,0,-1,-1])]):
                for n in gs['g']:
                    #can build city if player has road, there are no blocking roads, and there are not adjacent settlements/cities
                    if gs['g'][n]['s'] == None and self.adj_nodes_free(n,gs) and self.no_blocking_rds(n,gs) and self.has_road_access(n,player,gs):
                        am_sett.append(n)
        
        ##can afford road
        if gs['p'][player]['pieces']['r'] >= 1:
            if all([i >= 0 for i in add_vec(copy.deepcopy(gs['p'][player]['hand']),[0,-1,0,-1,0])]):
                for n in gs['g']:
                    #if player has a road to this location
                    if player in [gs['g'][n]['e'][e] for e in gs['g'][n]['e']]:
                        #get available onward roads
                        for e in gs['g'][n]['e']:
                            if gs['g'][n]['e'][e] == None:
                                am_road.append([e,n])

        moves = []
        while am_city:
            pm_gs = self.place_city(player,am_city.pop(),gs)
            moves.append(State(gs=pm_gs,this_player=player,parent=self))

        while am_sett:
            pm_gs = self.place_settlement(player,am_sett.pop(),gs)
            moves.append(State(gs=pm_gs,this_player=player,parent=self))

        while am_road:
            road_to_build = am_road.pop()
            pm_gs = self.place_road(player, road_to_build[0], road_to_build[1], gs)
            pm_gs = self.pay_for_purchase(pm_gs,player, PRICES['road'])            
            pm_gs = self.longest_road_check(pm_gs,player)
            moves.append(State(gs=pm_gs,this_player=player,parent=self))


        bank_trade_moves = self.get_bank_trade_moves(gs,player)

        for i in range(len(gs['p'][player]['hand'])):
            if gs['p'][player]['hand'][i] >= 4 and len(bank_trade_moves) < 1:
                raise Exception('bank move problem')


        while bank_trade_moves:
            use_to_purchase = bank_trade_moves.pop()
            for card_to_buy in list(RMAP.keys()):
                pm_gs = self.make_bank_trade(gs,player,use_to_purchase,card_to_buy)
                moves.append(State(gs=pm_gs,this_player=player,parent=self))

        port_trade_moves = self.get_port_trade_moves(gs,player)
        
        while port_trade_moves:
            port_move = port_trade_moves.pop()
            for card_to_buy in list(RMAP.keys()):
                pm_gs = self.make_port_trade(gs,player,port_move, card_to_buy)
                moves.append(State(gs=pm_gs,this_player=player,parent=self))
                
        ##buy dev cards

        can_buy_dev_card = self.can_buy_dev_card(gs,player)
        if can_buy_dev_card == True:
            pm_gs = self.buy_dev_card(gs,player)
            moves.append(State(gs=pm_gs,this_player=player,parent=self))

         ## if a dc card hasn't been played
        if not self.played_dc:
            if gs['p'][player]['dev_hand']['knight'] >= 1:
                #can play knight card
                pass
                #print('can play knight')

            if gs['p'][player]['dev_hand']['road_building'] >= 1:
                #player must have 2 roads in order to use road building card
                if gs['p'][player]['pieces']['r'] >= 2:
                    #build road cards
                    available_roads = []
                    for n in gs['g']:
                        #if player has a road to this location
                        if player in [gs['g'][n]['e'][e] for e in gs['g'][n]['e']]:
                            #get available onward roads
                            for e in gs['g'][n]['e']:
                                if gs['g'][n]['e'][e] == None:
                                    available_roads.append([e,n])
                    combos = [[x,y] for x in available_roads for y in available_roads]
                    for c in combos:
                        pm_gs=copy.deepcopy(gs)
                        road1 = c[0]                
                        pm_gs = self.place_road(player, road1[0], road1[1], pm_gs)
                        road2 = c[1]                
                        pm_gs = self.place_road(player, road2[0], road2[1], pm_gs)
                        pm_gs['p'][player]['dev_hand']['road_building'] -= 1
                        pm_gs = self.longest_road_check(pm_gs,player)
                        moves.append(State(gs=pm_gs,this_player=player,parent=self,played_dc=True))

            
            if gs['p'][player]['dev_hand']['monopoly'] >= 1:
                # monopoly card
                resource_list = list(RMAP.keys())
                for idx, val in enumerate(resource_list):
                    pm_gs=copy.deepcopy(gs)
                    if player == 'A':
                        other_player='B'
                    else:
                        other_player='A'
                    opponent_cards = gs['p'][other_player]['hand'][idx]
                    pm_gs['p'][other_player]['hand'][idx] = 0
                    pm_gs['p'][player]['hand'][idx] += opponent_cards
                    pm_gs['p'][player]['dev_hand']['monopoly'] -= 1
                    moves.append(State(gs=pm_gs,this_player=player,parent=self,played_dc=True))
            
            if gs['p'][player]['dev_hand']['year_plenty'] >= 1:
                # Year of plenty
                combos = [[x,y] for x in list(RMAP.keys()) for y in list(RMAP.keys())]
                for c in combos:
                    pm_gs=copy.deepcopy(gs)
                    card_type1 =c[0]
                    pm_gs = self.collect_card(pm_gs,player,card_type1)
                    card_type2 = c[1]
                    pm_gs = self.collect_card(pm_gs,player,card_type2)
                    pm_gs['p'][player]['dev_hand']['year_plenty'] -= 1
                    moves.append(State(gs=pm_gs,this_player=player,parent=self,played_dc=True))

            if gs['p'][player]['dev_hand']['victory_point'] >= 1:
                #victory card
                pm_gs=copy.deepcopy(gs)
                pm_gs['p'][player]['v']+=1
                pm_gs['p'][player]['dev_hand']['victory_point'] -= 1
                moves.append(State(gs=pm_gs,this_player=player,parent=self,played_dc=True))

        rmoves = []
        for s in moves:
            points = s.gs['p'][player]['v']
            h = self.calc_heuristic(s.gs,player)
            rmoves.append({'s':s,'v':h})
        return rmoves


    def calc_heuristic(self,gs,player):

        h = 0
        for n in gs['g']:
            dots = self.get_node_total_dots(gs,n)
            
            if gs['g'][n]['c'] == player:
                h += (100 * dots)
            if gs['g'][n]['s'] == player:
                h += (50 * dots) 
            for e in gs['g'][n]['e']:
                if gs['g'][n]['e'][e] == player:
                    #discourage building road to occupied node
                    if gs['g'][n]['s'] != None or gs['g'][n]['c'] != None:
                        h -= (20 * dots)
                    #discourage building road to node blocked by roads
                    elif not self.no_blocking_rds(e,gs):
                        h -= (20 * dots)
                    #is road to place where settlement can be built
                    elif gs['g'][n]['s'] == None and \
                        gs['g'][n]['c'] == None and \
                        self.adj_nodes_free(e,gs) and \
                        self.no_blocking_rds(e,gs):
                        h += (40 * dots)

        return h


    def get_node_total_dots(self,gs,n):
        dot_vals = [gs['b'][ni]['sc'] for ni in [ ','.join(np.split(',')[:3]) for np in n.split(';')]]
        return sum([DOTS[i] for i in dot_vals])

    def can_buy_dev_card(self,gs,player):
        #can afford dev card 
        if len(gs['dev_deck']) == 0:
            return False
        if all([i >= 0 for i in add_vec(copy.deepcopy(gs['p'][player]['hand']),[-1,0,-1,0,-1])]):
            return True
        return False

    def buy_dev_card(self,gs,player):
        gs = copy.deepcopy(gs)
        price = [-1,0,-1,0,-1]
        gs = self.pay_for_purchase(gs,player, price)
        ind=gs['dev_deck'].pop()
        gs['p'][player]['dev_hand'][ind]+=1
        return gs


    def start_turn(self):
        roll = roll_dice()
        self.collect_resource_cards(roll)

    def next_turn(self):
        self.gs['turn'] = (self.gs['turn'] + 1) % len(self.players)
        #print( 'now its {} turn '.format(self.gs['turn']) )


    def collect_resource_cards(self,roll):
        for player in ['A','B']:
            self.gs['p'][player]['hand'] = add_vec(copy.deepcopy(self.gs['p'][player]['hand']),copy.deepcopy(self.gs['p'][player][str(roll)]))
        return 


    def calc_result(self):
        return [p for p in self.gs['p'] if self.gs['p'][p]['v'] >= 10][0]

    def is_game_over(self):
        for p in ['A','B']:            
            if self.gs['p'][p]['v'] >= 10:
                return True
        return False
        

    


class Game(object):
    def __init__(self,gs):
        
        self.gs = gs
        self.players = ['A','B']


    def show(self):
        print( 'showing game info\n' )
        pp.pprint(self.gs.gs)        

    def get_winner(self):
        print('game find winner scores: A: {} B: {}'.format( self.gs.gs['p']['A']['v'],self.gs.gs['p']['B']['v'] ))
        for p in self.gs.gs['p']:
            if self.gs.gs['p'][p]['v'] >= 10:
                return p 
        return None

    def adjust_victory_points(self,gs,player,points):
        gs.gs['p'][player]['v'] += points

    def next_game_turn(self):
        self.gs.gs['turn'] = (self.gs.gs['turn'] + 1) % len(self.gs.players) 


    def adj_nodes_free(self,n,gs):
        # adjacent nodes do not have settlement or city  
        if any([ gs.gs['g'][e]['s'] != None or gs.gs['g'][e]['c'] != None for e in gs.gs['g'][n]['e']]):
            return False
        return True

    def no_blocking_rds(self,n,gs):
        # 2 or more edges not occupied by opponent roads
        this_player = self.players[self.gs.gs['turn']]
        if len([e for e in gs.gs['g'][n]['e'] if gs.gs['g'][n]['e'][e] != None  and gs.gs['g'][n]['e'][e] != this_player]) >= 2:
            return False
        return True

    def get_node_total_dots(self,gs,n):
            dot_vals = [gs['b'][ni]['sc'] for ni in [ ','.join(np.split(',')[:3]) for np in n.split(';')]]
            return sum([DOTS[i] for i in dot_vals])

    def get_available_moves_ip(self,gs):
        #available moves for initial placement phase as start of game
        am = []        
        for n in gs.gs['g']:
            if gs.gs['g'][n]['s'] == None and self.adj_nodes_free(n,gs) and self.no_blocking_rds(n,gs):
                roll_vals = self.get_node_total_dots(gs.gs,n)                
                am.append({'n':n, 'rv':roll_vals})
        print( '{} available moves '.format( len(am) ) )
        return am



    def place_settlement(self,player,n,gs, redeem_resource_cards=False):
        gs.gs['g'][n]['s'] = player
        self.update_role_val_game_state(player,n,gs,redeem_resource_cards=redeem_resource_cards)
        self.adjust_victory_points(gs, player,1)
        gs.gs['p'][player]['pieces']['s'] -= 1
        return gs.gs

    def place_city(self,player,n,gs):
        assert(gs.gs['g'][n]['s'] == player   ) 
        gs.gs['g'][n]['c'] = player
        self.update_role_val_game_state(player,n,gs)
        self.adjust_victory_points(gs, player,2)
        gs.gs['p'][player]['pieces']['c'] -= 1
        gs.gs['p'][player]['pieces']['s'] += 1
        return gs

    def collect_card(self,gs,card_type):
        gs.gs['p'][player]['hand'] = add_vec(gs.gs['p'][player]['hand'],RMAP[card_type])


    def update_role_val_game_state(self,player,n,gs,redeem_resource_cards=False):
        for t in n.split(';'):
            t = ','.join(t.split(',')[:3])
            card_type = gs.gs['b'][t]['r']
            #update roll map for player 
            roll_val = gs.gs['b'][t]['sc']         
            if roll_val != '0':
                gs.gs['p'][player][roll_val] = add_vec(gs.gs['p'][player][roll_val],RMAP[card_type])
            #add card to hand, only not true on first of 2 initial placements 
            if redeem_resource_cards:
                if card_type != 'Desert':
                    gs.gs['p'][player]['hand'] = add_vec(gs.gs['p'][player]['hand'],RMAP[card_type])

    def place_road(self,player,n,dest,gs):
        gs.gs['g'][n]['e'][dest] = player
        gs.gs['g'][dest]['e'][n] = player
        gs.gs['p'][player]['pieces']['r'] -= 1
        assert( gs.gs['g'][n]['e'][dest] == gs.gs['g'][dest]['e'][n]  ) 
        return gs.gs

    def available_roads_from_sett(self,n,gs):
        ar = []
        for e in gs.gs['g'][n]['e']:
            #from settlement, is road to next settlement available
            if gs.gs['g'][n]['e'][e] == None:
                ar.append(e)
        return ar 

    def build_random_road(self,player,n,gs):
        ar = self.available_roads_from_sett(n,gs)
        rd = ar[random.randrange(len(ar))] #get random edge from node to build road
        gs = self.place_road(player,n,rd,gs)
        return gs

    def initial_placement(self,gs):
        print('initial_placement ')
        pl = [p for p in self.players]
        pturns = [pl,list(reversed(pl))]
        # this is currently random placement ....
        while pturns[0]:
            this_player = pturns[0].pop(0)
            am = self.get_available_moves_ip(gs)
            best_n = max(am, key=lambda x:x['rv'])
            print( 'best_n rv {}'.format(best_n['rv']) )
            n = best_n['n']
            gs.gs = self.place_settlement(this_player,n,gs)
            gs.gs = self.build_random_road(this_player,n,gs)
            
        while pturns[1]:
            this_player = pturns[1].pop(0)
            am = self.get_available_moves_ip(gs)
            best_n = max(am, key=lambda x:x['rv'])
            print( 'best_n rv {}'.format(best_n['rv']) )
            n = best_n['n']
            gs.gs = self.place_settlement(this_player,n,gs,redeem_resource_cards=True)
            gs.gs = self.build_random_road(this_player,n,gs)
        return gs

    def collect_resource_cards(self,roll):
        
        for player in ['A','B']:
            #print("{} before collecting cards: {}".format( player  , self.gs.gs['p'][player]['hand'] ))
            self.gs.gs['p'][player]['hand'] = add_vec(self.gs.gs['p'][player]['hand'],self.gs.gs['p'][player][str(roll)])
            #print("{} after collecting cards: {}".format( player  , self.gs.gs['p'][player]['hand'] ))

        return 

class MCTSNode(object):

    def __init__(self, state, parent=None):
        """
        Parameters
        ----------
        state : mctspy.games.common.TwoPlayersAbstractGameState
        parent : MonteCarloTreeSearchNode
        """
        self.state = state
        self.parent = parent
        self.children = []

    @property
    def untried_actions(self):
        pass

    @property
    def q(self):
        pass

    @property
    def n(self):
        pass

    def expand(self):
        pass

    def is_terminal_node(self):
        pass

    def rollout(self):
        pass

    def backpropagate(self, reward):
        pass

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        if not self.children:
            return None
        
        choices_weights = [
            (c.q / c.n) + c_param * np.sqrt((2 * np.log(self.n) / c.n))
            for c in self.children
        ]
        print( 'choices_weights --> ', choices_weights )
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):        
        if not possible_moves:
            return None
        best_move = max(possible_moves, key=lambda x:x['v'])
        return best_move['s']


class TwoPlayersGameMCTSNode(MCTSNode):
    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self._number_of_visits = 0.
        self._results = {'A':0,'B':0}
        self._untried_actions = None


    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.get_available_moves() #legal actions
        return self._untried_actions

    @property
    def q(self):
        wins = self._results[self.state.players[self.state.gs['turn'] ]]
        loses = self._results[self.state.players[ (self.state.gs['turn'] + 1) % len(self.state.players) ]]
        return wins - loses

    @property
    def n(self):
        return self._number_of_visits

    def expand(self):
        next_state = self.untried_actions.pop()
        child_node = TwoPlayersGameMCTSNode(
            state = next_state['s'], parent=self
        )
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            current_rollout_state.start_turn()
            playing_turn = True
            while playing_turn:
                possible_moves = current_rollout_state.get_available_moves() #legal actions
                best_next_move = self.rollout_policy(possible_moves)
                if best_next_move == None:
                    playing_turn = False
                    break
                else:
                    current_rollout_state = best_next_move
            current_rollout_state.next_turn()
        result = current_rollout_state.calc_result()
        return result

    def backpropagate(self, result, count):
        self._number_of_visits += 1.
        self._results[result] += 1.
        count += 1
        if self.parent:
            self.parent.backpropagate(result,count)


class MCTS(object):
    def __init__(self,node):
        self.root = node       

    def best_action(self,simulations_number):
        for c in range(0, simulations_number):            
            v = self.tree_policy()
            if not v:
                continue
            rollout_winner = v.rollout()
            v.backpropagate(rollout_winner,0)
        return self.root.best_child(c_param=0.)

    def tree_policy(self):
        """ selects node to run rollout/playout for """
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
                if not current_node:
                    return None
            
        return current_node

def play(g):
    start = time.time()
    g.gs = g.initial_placement(g.gs)
    print( 'after initial placement' , g.gs.gs['turn'])  
    root = TwoPlayersGameMCTSNode(state = g.gs)

    print('best action state ', root.state)
    print( 'after assigment ' , root.state.gs['turn'])  

    game_counter = 0

    complete = False
    winner = []
    while not complete:     

        game_counter += 1
        print('GAME COUNTER ', game_counter)   
        player_turn = g.gs.gs['turn']   
        assert( g.gs.gs['turn'] == root.state.gs['turn'])  
        roll = roll_dice()            
        g.collect_resource_cards(roll)
        for p in ['A','B']:
            root.state.gs['p'][p]['hand'] = g.gs.gs['p'][p]['hand']
        playing_turn_count = 0
        playing_turn = True
        while playing_turn:
            print('play turn for  {} A: {} B: {}'.format( g.gs.gs['turn'], g.gs.gs['p']['A']['hand'], g.gs.gs['p']['B']['hand']))
            playing_turn_count += 1
            mcts = MCTS(root)
            best_action = mcts.best_action(25)
            if best_action == None:
                playing_turn = False
                root = TwoPlayersGameMCTSNode(state = mcts.root.state)
                break

            print('mcts found best action for player {} '.format(g.gs.gs['turn']))
            g.gs = best_action.state
            assert( best_action.state.gs['p']['A']['v'] == g.gs.gs['p']['A']['v'] )
            assert( best_action.state.gs['p']['B']['v'] == g.gs.gs['p']['B']['v'] )
            root = TwoPlayersGameMCTSNode(state = best_action.state)
            root.state.gs['turn'] = g.gs.gs['turn'] 
            
            if playing_turn_count == 5:
                playing_turn = False
                break
 
        print('on to next turn')
        winner = g.get_winner()

        if winner:
            complete = True

        g.gs.gs['turn'] = player_turn 
        g.next_game_turn()
        root.state.gs['turn'] = g.gs.gs['turn']  

    end = time.time()
    time_elapsed = end - start
    return winner, num_moves, time_elapsed

h = Hexgrid(2)
p = [Player("Alice"),Player("Bob")]
b = Board(start_cube='-2,0,2',hex_grid=h)
s = State(hex_grid=h,board=b,players=p)
g = Game(gs=s)
w,m,t = play(g=g)
print('Winner is {} in {} moves and elapsed time {}'.format(w,m,t))
