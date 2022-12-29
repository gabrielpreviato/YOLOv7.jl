using Flux

using YOLOv7: Node

# custom split layer
struct Split{T}
  paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

struct YOLOChain
    nodes::Vector{Node}
    components::Vector{Any}
    # results::Vector{AbstractArray}
    e::Expr
end

function YOLOChain(nodes::Vector{Node}, components::Vector{Any})
    # results::Vector{AbstractArray}
    x = rand(Float32, 160, 160, 3, 1)
    e = _generate_chain(Tuple(components), Tuple(nodes), x)

    return YOLOChain(nodes, components, e)
end

Flux.@functor YOLOChain

function Flux.trainable(m::YOLOChain)
    (m.components,)
end

# function YOLOChain(nodes::Vector{Node}, components::Vector{Any})
#     # results = []
#     # x = randn(Float32, 640, 640, 3, 1)
#     # for (node, component) in zip(nodes, components)
#     #     if length(node.parents) == 0
#     #         push!(results, x)
#     #     else
#     #         # println(node)
#     #         # println(component)
#     #         # println(node.op[2])
#     #         # println(size(results[node.op[2][1]]))
#     #         push!(results, component(results[node.op[2]]...))
#     #     end
#     #     # println(length(results))
#     # end

#     return YOLOChain(nodes, components)
# end

Base.getindex(c::YOLOChain, i::Int64) = YOLOChain(c.nodes[i], c.components[i])

Base.getindex(c::YOLOChain, i::UnitRange{Int64}) = YOLOChain(c.nodes[i], c.components[i])

(m::YOLOChain)(x::AbstractArray) = _apply_chain(Tuple(m.components), Tuple(m.nodes), x)
# (m::YOLOChain)(x::AbstractArray) = _apply_chain(Tuple(m.components), x)
# (m::YOLOChain)(x::AbstractArray) = _apply_chain(Tuple(m.components), m.e, x)

@generated function _apply_chain(layers::Tuple{Vararg{<:Any,N}}, e, x) where {N}
    Expr(:block, e...)
end

function _chain(m, node, component, x::AbstractArray)
    # println(node)
    if length(node.parents) == 0
        return x
    end

    return component([_chain(m, m.nodes[i], m.components[i], x) for i in node.op[2]]...)
    # return _chain(m, m.nodes[i], m.components[i], component(x))
end

@generated function _apply_chain(layers::Tuple{Vararg{<:Any,N}}, x) where {N}
    symbols = vcat(:x, [gensym() for _ in 1:N])
    calls = [:($(symbols[i+1]) = layers[$i]($(symbols[i]))) for i in 1:N]
    # println(symbols)
    println(symbols,"\n",calls)
    Expr(:block, calls...)
  end

# @eval function _apply_chain(layers::Tuple{Vararg{<:Any,N}}, nodes::Tuple{Vararg{<:Any,N}}, x) where {N}
#     # println(m)
#     symbols = vcat(:x, [gensym() for _ in 1:N])
#     froms = [n.op[2] for n in nodes]

#     # calls_from = [:($(froms[i]) = ((nodes[$i]).op[2])) for i in 1:N]
#     # println(Expr(:block, calls_from...))
#     # calls = [:($(symbols[i+1]) = layers[$i]((symbols[$(nodes[i])]))) for i in 1:N]
#     # println(nodes[1].op[2])
#     calls = [:($(symbols[i+1]) = layers[$i]($(symbols[(f...)]))) for (i, f, s) in zip(1:N, froms, symbols)]
#     # println(calls)
#     println(symbols,"\n",calls)
#     eval(:(x = $x))
#     eval(:(layers = $layers))
#     eval(:(symbols = $symbols))
#     eval.(calls)
# end
@generated function _apply_chain(layers::Tuple{Vararg{<:Any,N}}, nodes::Tuple{Vararg{<:Any,N}}, x) where {N}
    # quote
    #     var"##736" = (layers[1])(x)
    #     var"##737" = (layers[2])(var"##736")
    #     var"##738" = (layers[3])(var"##737")
    #     var"##739" = (layers[4])(var"##738")
    #     var"##740" = (layers[5])(var"##739")
    #     var"##741" = (layers[6])(var"##739")
    #     var"##742" = (layers[7])(var"##741")
    #     var"##743" = (layers[8])(var"##742")
    #     var"##744" = (layers[9])(var"##743")
    #     var"##745" = (layers[10])(var"##744")
    #     var"##842" = var"##745"
    #     var"##843" = cat((var"##842", var"##743")...; dims = 3)
    #     var"##844" = cat((var"##843", var"##741")...; dims = 3)
    #     var"##845" = cat((var"##844", var"##740")...; dims = 3)
    #     var"##746" = var"##845"
    #     var"##747" = (layers[12])(var"##746")
    #     var"##748" = (layers[13])(var"##747")
    #     var"##749" = (layers[14])(var"##748")
    #     var"##750" = (layers[15])(var"##747")
    #     var"##751" = (layers[16])(var"##750")
    #     var"##846" = var"##751"
    #     var"##847" = cat((var"##846", var"##749")...; dims = 3)
    #     var"##752" = var"##847"
    #     var"##753" = (layers[18])(var"##752")
    #     var"##754" = (layers[19])(var"##752")
    #     var"##755" = (layers[20])(var"##754")
    #     var"##756" = (layers[21])(var"##755")
    #     var"##757" = (layers[22])(var"##756")
    #     var"##758" = (layers[23])(var"##757")
    #     var"##848" = var"##758"
    #     var"##849" = cat((var"##848", var"##756")...; dims = 3)
    #     var"##850" = cat((var"##849", var"##754")...; dims = 3)
    #     var"##851" = cat((var"##850", var"##753")...; dims = 3)
    #     var"##759" = var"##851"
    #     var"##760" = (layers[25])(var"##759")
    #     var"##761" = (layers[26])(var"##760")
    #     var"##762" = (layers[27])(var"##761")
    #     var"##763" = (layers[28])(var"##760")
    #     var"##764" = (layers[29])(var"##763")
    #     var"##852" = var"##764"
    #     var"##853" = cat((var"##852", var"##762")...; dims = 3)
    #     var"##765" = var"##853"
    #     var"##766" = (layers[31])(var"##765")
    #     var"##767" = (layers[32])(var"##765")
    #     var"##768" = (layers[33])(var"##767")
    #     var"##769" = (layers[34])(var"##768")
    #     var"##770" = (layers[35])(var"##769")
    #     var"##771" = (layers[36])(var"##770")
    #     var"##854" = var"##771"
    #     var"##855" = cat((var"##854", var"##769")...; dims = 3)
    #     var"##856" = cat((var"##855", var"##767")...; dims = 3)
    #     var"##857" = cat((var"##856", var"##766")...; dims = 3)
    #     var"##772" = var"##857"
    #     var"##773" = (layers[38])(var"##772")
    #     var"##774" = (layers[39])(var"##773")
    #     var"##775" = (layers[40])(var"##774")
    #     var"##776" = (layers[41])(var"##773")
    #     var"##777" = (layers[42])(var"##776")
    #     var"##858" = var"##777"
    #     var"##859" = cat((var"##858", var"##775")...; dims = 3)
    #     var"##778" = var"##859"
    #     var"##779" = (layers[44])(var"##778")
    #     var"##780" = (layers[45])(var"##778")
    #     var"##781" = (layers[46])(var"##780")
    #     var"##782" = (layers[47])(var"##781")
    #     var"##783" = (layers[48])(var"##782")
    #     var"##784" = (layers[49])(var"##783")
    #     var"##860" = var"##784"
    #     var"##861" = cat((var"##860", var"##782")...; dims = 3)
    #     var"##862" = cat((var"##861", var"##780")...; dims = 3)
    #     var"##863" = cat((var"##862", var"##779")...; dims = 3)
    #     var"##785" = var"##863"
    #     var"##786" = (layers[51])(var"##785")
    #     var"##787" = (layers[52])(var"##786")
    #     var"##788" = (layers[53])(var"##787")
    #     var"##789" = (layers[54])(var"##788")
    #     var"##790" = (layers[55])(var"##771")
    #     var"##864" = var"##790"
    #     var"##865" = cat((var"##864", var"##789")...; dims = 3)
    #     var"##791" = var"##865"
    #     var"##792" = (layers[57])(var"##791")
    #     var"##793" = (layers[58])(var"##791")
    #     var"##794" = (layers[59])(var"##793")
    #     var"##795" = (layers[60])(var"##794")
    #     var"##796" = (layers[61])(var"##795")
    #     var"##797" = (layers[62])(var"##796")
    #     var"##866" = var"##797"
    #     var"##867" = cat((var"##866", var"##796")...; dims = 3)
    #     var"##868" = cat((var"##867", var"##795")...; dims = 3)
    #     var"##869" = cat((var"##868", var"##794")...; dims = 3)
    #     var"##870" = cat((var"##869", var"##793")...; dims = 3)
    #     var"##871" = cat((var"##870", var"##792")...; dims = 3)
    #     var"##798" = var"##871"
    #     var"##799" = (layers[64])(var"##798")
    #     var"##800" = (layers[65])(var"##799")
    #     var"##801" = (layers[66])(var"##800")
    #     var"##802" = (layers[67])(var"##758")
    #     var"##872" = var"##802"
    #     var"##873" = cat((var"##872", var"##801")...; dims = 3)
    #     var"##803" = var"##873"
    #     var"##804" = (layers[69])(var"##803")
    #     var"##805" = (layers[70])(var"##803")
    #     var"##806" = (layers[71])(var"##805")
    #     var"##807" = (layers[72])(var"##806")
    #     var"##808" = (layers[73])(var"##807")
    #     var"##809" = (layers[74])(var"##808")
    #     var"##874" = var"##809"
    #     var"##875" = cat((var"##874", var"##808")...; dims = 3)
    #     var"##876" = cat((var"##875", var"##807")...; dims = 3)
    #     var"##877" = cat((var"##876", var"##806")...; dims = 3)
    #     var"##878" = cat((var"##877", var"##805")...; dims = 3)
    #     var"##879" = cat((var"##878", var"##804")...; dims = 3)
    #     var"##810" = var"##879"
    #     var"##811" = (layers[76])(var"##810")
    #     var"##812" = (layers[77])(var"##811")
    #     var"##813" = (layers[78])(var"##812")
    #     var"##814" = (layers[79])(var"##811")
    #     var"##815" = (layers[80])(var"##814")
    #     var"##880" = var"##815"
    #     var"##881" = cat((var"##880", var"##813")...; dims = 3)
    #     var"##882" = cat((var"##881", var"##797")...; dims = 3)
    #     var"##816" = var"##882"
    #     var"##817" = (layers[82])(var"##816")
    #     var"##818" = (layers[83])(var"##816")
    #     var"##819" = (layers[84])(var"##818")
    #     var"##820" = (layers[85])(var"##819")
    #     var"##821" = (layers[86])(var"##820")
    #     var"##822" = (layers[87])(var"##821")
    #     var"##883" = var"##822"
    #     var"##884" = cat((var"##883", var"##821")...; dims = 3)
    #     var"##885" = cat((var"##884", var"##820")...; dims = 3)
    #     var"##886" = cat((var"##885", var"##819")...; dims = 3)
    #     var"##887" = cat((var"##886", var"##818")...; dims = 3)
    #     var"##888" = cat((var"##887", var"##817")...; dims = 3)
    #     var"##823" = var"##888"
    #     var"##824" = (layers[89])(var"##823")
    #     var"##825" = (layers[90])(var"##824")
    #     var"##826" = (layers[91])(var"##825")
    #     var"##827" = (layers[92])(var"##824")
    #     var"##828" = (layers[93])(var"##827")
    #     var"##889" = var"##828"
    #     var"##890" = cat((var"##889", var"##826")...; dims = 3)
    #     var"##891" = cat((var"##890", var"##785")...; dims = 3)
    #     var"##829" = var"##891"
    #     var"##830" = (layers[95])(var"##829")
    #     var"##831" = (layers[96])(var"##829")
    #     var"##832" = (layers[97])(var"##831")
    #     var"##833" = (layers[98])(var"##832")
    #     var"##834" = (layers[99])(var"##833")
    #     var"##835" = (layers[100])(var"##834")
    #     var"##892" = var"##835"
    #     var"##893" = cat((var"##892", var"##834")...; dims = 3)
    #     var"##894" = cat((var"##893", var"##833")...; dims = 3)
    #     var"##895" = cat((var"##894", var"##832")...; dims = 3)
    #     var"##896" = cat((var"##895", var"##831")...; dims = 3)
    #     var"##897" = cat((var"##896", var"##830")...; dims = 3)
    #     var"##836" = var"##897"
    #     var"##837" = (layers[102])(var"##836")
    #     var"##838" = (layers[103])(var"##837")
    #     var"##839" = (layers[104])(var"##838")
    #     var"##840" = (layers[105])(var"##839")
    #     var"##841" = (layers[106])(var"##840")
    # end
    # quote
    #     var"##1313" = (layers[1])(x)
    #     var"##1314" = (layers[2])(var"##1313")
    #     var"##1315" = (layers[3])(var"##1314")
    #     var"##1316" = (layers[4])(var"##1315")
    #     var"##1317" = (layers[5])(var"##1316")
    #     var"##1318" = (layers[6])(var"##1316")
    #     var"##1319" = (layers[7])(var"##1318")
    #     var"##1320" = (layers[8])(var"##1319")
    #     var"##1321" = (layers[9])(var"##1320")
    #     var"##1322" = (layers[10])(var"##1321")
    #     var"##1404" = var"##1322"
    #     var"##1405" = cat((var"##1404", var"##1320")...; dims = 3)
    #     var"##1406" = cat((var"##1405", var"##1318")...; dims = 3)
    #     var"##1407" = cat((var"##1406", var"##1317")...; dims = 3)
    #     var"##1323" = var"##1407"
    #     var"##1324" = (layers[12])(var"##1323")
    #     var"##1325" = (layers[13])(var"##1324")
    #     var"##1326" = (layers[14])(var"##1325")
    #     var"##1327" = (layers[15])(var"##1324")
    #     var"##1328" = (layers[16])(var"##1327")
    #     var"##1408" = var"##1328"
    #     var"##1409" = cat((var"##1408", var"##1326")...; dims = 3)
    #     var"##1329" = var"##1409"
    #     var"##1330" = (layers[18])(var"##1329")
    #     var"##1331" = (layers[19])(var"##1329")
    #     var"##1332" = (layers[20])(var"##1331")
    #     var"##1333" = (layers[21])(var"##1332")
    #     var"##1334" = (layers[22])(var"##1333")
    #     var"##1335" = (layers[23])(var"##1334")
    #     var"##1410" = var"##1335"
    #     var"##1411" = cat((var"##1410", var"##1333")...; dims = 3)
    #     var"##1412" = cat((var"##1411", var"##1331")...; dims = 3)
    #     var"##1413" = cat((var"##1412", var"##1330")...; dims = 3)
    #     var"##1336" = var"##1413"
    #     var"##1337" = (layers[25])(var"##1336")
    #     var"##1338" = (layers[26])(var"##1337")
    #     var"##1339" = (layers[27])(var"##1338")
    #     var"##1340" = (layers[28])(var"##1337")
    #     var"##1341" = (layers[29])(var"##1340")
    #     var"##1414" = var"##1341"
    #     var"##1415" = cat((var"##1414", var"##1339")...; dims = 3)
    #     var"##1342" = var"##1415"
    #     var"##1343" = (layers[31])(var"##1342")
    #     var"##1344" = (layers[32])(var"##1342")
    #     var"##1345" = (layers[33])(var"##1344")
    #     var"##1346" = (layers[34])(var"##1345")
    #     var"##1347" = (layers[35])(var"##1346")
    #     var"##1348" = (layers[36])(var"##1347")
    #     var"##1416" = var"##1348"
    #     var"##1417" = cat((var"##1416", var"##1346")...; dims = 3)
    #     var"##1418" = cat((var"##1417", var"##1344")...; dims = 3)
    #     var"##1419" = cat((var"##1418", var"##1343")...; dims = 3)
    #     var"##1349" = var"##1419"
    #     var"##1350" = (layers[38])(var"##1349")
    #     var"##1351" = (layers[39])(var"##1350")
    #     var"##1352" = (layers[40])(var"##1351")
    #     var"##1353" = (layers[41])(var"##1352")
    #     var"##1354" = (layers[42])(var"##1336")
    #     var"##1420" = var"##1354"
    #     var"##1421" = cat((var"##1420", var"##1353")...; dims = 3)
    #     var"##1355" = var"##1421"
    #     var"##1356" = (layers[44])(var"##1355")
    #     var"##1357" = (layers[45])(var"##1355")
    #     var"##1358" = (layers[46])(var"##1357")
    #     var"##1359" = (layers[47])(var"##1358")
    #     var"##1360" = (layers[48])(var"##1359")
    #     var"##1361" = (layers[49])(var"##1360")
    #     var"##1422" = var"##1361"
    #     var"##1423" = cat((var"##1422", var"##1360")...; dims = 3)
    #     var"##1424" = cat((var"##1423", var"##1359")...; dims = 3)
    #     var"##1425" = cat((var"##1424", var"##1358")...; dims = 3)
    #     var"##1426" = cat((var"##1425", var"##1357")...; dims = 3)
    #     var"##1427" = cat((var"##1426", var"##1356")...; dims = 3)
    #     var"##1362" = var"##1427"
    #     var"##1363" = (layers[51])(var"##1362")
    #     var"##1364" = (layers[52])(var"##1363")
    #     var"##1365" = (layers[53])(var"##1336")
    #     var"##1428" = var"##1365"
    #     var"##1429" = cat((var"##1428", var"##1364")...; dims = 3)
    #     var"##1366" = var"##1429"
    #     var"##1367" = (layers[55])(var"##1366")
    #     var"##1368" = (layers[56])(var"##1366")
    #     var"##1369" = (layers[57])(var"##1368")
    #     var"##1370" = (layers[58])(var"##1369")
    #     var"##1371" = (layers[59])(var"##1370")
    #     var"##1372" = (layers[60])(var"##1371")
    #     var"##1430" = var"##1372"
    #     var"##1431" = cat((var"##1430", var"##1371")...; dims = 3)
    #     var"##1432" = cat((var"##1431", var"##1370")...; dims = 3)
    #     var"##1433" = cat((var"##1432", var"##1369")...; dims = 3)
    #     var"##1434" = cat((var"##1433", var"##1368")...; dims = 3)
    #     var"##1435" = cat((var"##1434", var"##1367")...; dims = 3)
    #     var"##1373" = var"##1435"
    #     var"##1374" = (layers[62])(var"##1373")
    #     var"##1375" = (layers[63])(var"##1374")
    #     var"##1376" = (layers[64])(var"##1374")
    #     var"##1377" = (layers[65])(var"##1376")
    #     var"##1436" = var"##1377"
    #     var"##1437" = cat((var"##1436", var"##1375")...; dims = 3)
    #     var"##1438" = cat((var"##1437", var"##1361")...; dims = 3)
    #     var"##1378" = var"##1438"
    #     var"##1379" = (layers[67])(var"##1378")
    #     var"##1380" = (layers[68])(var"##1378")
    #     var"##1381" = (layers[69])(var"##1380")
    #     var"##1382" = (layers[70])(var"##1381")
    #     var"##1383" = (layers[71])(var"##1382")
    #     var"##1384" = (layers[72])(var"##1383")
    #     var"##1439" = var"##1384"
    #     var"##1440" = cat((var"##1439", var"##1383")...; dims = 3)
    #     var"##1441" = cat((var"##1440", var"##1382")...; dims = 3)
    #     var"##1442" = cat((var"##1441", var"##1381")...; dims = 3)
    #     var"##1443" = cat((var"##1442", var"##1380")...; dims = 3)
    #     var"##1444" = cat((var"##1443", var"##1379")...; dims = 3)
    #     var"##1385" = var"##1444"
    #     var"##1386" = (layers[74])(var"##1385")
    #     var"##1387" = (layers[75])(var"##1386")
    #     var"##1388" = (layers[76])(var"##1387")
    #     var"##1389" = (layers[77])(var"##1386")
    #     var"##1390" = (layers[78])(var"##1389")
    #     var"##1445" = var"##1390"
    #     var"##1446" = cat((var"##1445", var"##1388")...; dims = 3)
    #     var"##1447" = cat((var"##1446", var"##1349")...; dims = 3)
    #     var"##1391" = var"##1447"
    #     var"##1392" = (layers[80])(var"##1391")
    #     var"##1393" = (layers[81])(var"##1391")
    #     var"##1394" = (layers[82])(var"##1393")
    #     var"##1395" = (layers[83])(var"##1394")
    #     var"##1396" = (layers[84])(var"##1395")
    #     var"##1397" = (layers[85])(var"##1396")
    #     var"##1448" = var"##1397"
    #     var"##1449" = cat((var"##1448", var"##1396")...; dims = 3)
    #     var"##1450" = cat((var"##1449", var"##1395")...; dims = 3)
    #     var"##1451" = cat((var"##1450", var"##1394")...; dims = 3)
    #     var"##1452" = cat((var"##1451", var"##1393")...; dims = 3)
    #     var"##1453" = cat((var"##1452", var"##1392")...; dims = 3)
    #     var"##1398" = var"##1453"
    #     var"##1399" = (layers[87])(var"##1398")
    #     var"##1400" = (layers[88])(var"##1399")
    #     var"##1401" = (layers[89])(var"##1400")
    #     var"##1402" = (layers[90])(var"##1401")
    #     var"##1403" = (layers[91])(var"##1402")
    # end
    quote
        var"##320" = (layers[1])(x)
        var"##321" = (layers[2])(var"##320")
        var"##322" = (layers[3])(var"##321")
        var"##323" = (layers[4])(var"##322")
        var"##324" = (layers[5])(var"##323")
        var"##325" = (layers[6])(var"##323")
        var"##326" = (layers[7])(var"##325")
        var"##327" = (layers[8])(var"##326")
        var"##328" = (layers[9])(var"##327")
        var"##329" = (layers[10])(var"##328")
        var"##330" = cat(var"##329", var"##327", var"##325", var"##324"; dims = 3)
        var"##331" = (layers[12])(var"##330")
        var"##332" = (layers[13])(var"##331")
        var"##333" = (layers[14])(var"##332")
        var"##334" = (layers[15])(var"##331")
        var"##335" = (layers[16])(var"##334")
        var"##336" = cat(var"##335", var"##333"; dims = 3)
        var"##337" = (layers[18])(var"##336")
        var"##338" = (layers[19])(var"##336")
        var"##339" = (layers[20])(var"##338")
        var"##340" = (layers[21])(var"##339")
        var"##341" = (layers[22])(var"##340")
        var"##342" = (layers[23])(var"##341")
        var"##343" = cat(var"##342", var"##340", var"##338", var"##337"; dims = 3)
        var"##344" = (layers[25])(var"##343")
        var"##345" = (layers[26])(var"##344")
        var"##346" = (layers[27])(var"##345")
        var"##347" = (layers[28])(var"##344")
        var"##348" = (layers[29])(var"##347")
        var"##349" = cat(var"##348", var"##346"; dims = 3)
        var"##350" = (layers[31])(var"##349")
        var"##351" = (layers[32])(var"##349")
        var"##352" = (layers[33])(var"##351")
        var"##353" = (layers[34])(var"##352")
        var"##354" = (layers[35])(var"##353")
        var"##355" = (layers[36])(var"##354")
        var"##356" = cat(var"##355", var"##353", var"##351", var"##350"; dims = 3)
        var"##357" = (layers[38])(var"##356")
        var"##358" = (layers[39])(var"##357")
        var"##359" = (layers[40])(var"##358")
        var"##360" = (layers[41])(var"##359")
        var"##361" = (layers[42])(var"##343")
        var"##362" = cat(var"##361", var"##360"; dims = 3)
        var"##363" = (layers[44])(var"##362")
        var"##364" = (layers[45])(var"##362")
        var"##365" = (layers[46])(var"##364")
        var"##366" = (layers[47])(var"##365")
        var"##367" = (layers[48])(var"##366")
        var"##368" = (layers[49])(var"##367")
        var"##369" = cat(var"##368", var"##367", var"##366", var"##365", var"##364", var"##363"; dims = 3)
        var"##370" = (layers[51])(var"##369")
        var"##371" = (layers[52])(var"##370")
        var"##372" = (layers[53])(var"##343")
        var"##373" = cat(var"##372", var"##371"; dims = 3)
        var"##374" = (layers[55])(var"##373")
        var"##375" = (layers[56])(var"##373")
        var"##376" = (layers[57])(var"##375")
        var"##377" = (layers[58])(var"##376")
        var"##378" = (layers[59])(var"##377")
        var"##379" = (layers[60])(var"##378")
        var"##380" = cat(var"##379", var"##378", var"##377", var"##376", var"##375", var"##374"; dims = 3)
        var"##381" = (layers[62])(var"##380")
        var"##382" = (layers[63])(var"##381")
        var"##383" = (layers[64])(var"##381")
        var"##384" = (layers[65])(var"##383")
        var"##385" = cat(var"##384", var"##382", var"##368"; dims = 3)
        var"##386" = (layers[67])(var"##385")
        var"##387" = (layers[68])(var"##385")
        var"##388" = (layers[69])(var"##387")
        var"##389" = (layers[70])(var"##388")
        var"##390" = (layers[71])(var"##389")
        var"##391" = (layers[72])(var"##390")
        var"##392" = cat(var"##391", var"##390", var"##389", var"##388", var"##387", var"##386"; dims = 3)
        var"##393" = (layers[74])(var"##392")
        var"##394" = (layers[75])(var"##393")
        var"##395" = (layers[76])(var"##394")
        var"##396" = (layers[77])(var"##393")
        var"##397" = (layers[78])(var"##396")
        var"##398" = cat(var"##397", var"##395", var"##356"; dims = 3)
        var"##399" = (layers[80])(var"##398")
        var"##400" = (layers[81])(var"##398")
        var"##401" = (layers[82])(var"##400")
        var"##402" = (layers[83])(var"##401")
        var"##403" = (layers[84])(var"##402")
        var"##404" = (layers[85])(var"##403")
        var"##405" = cat(var"##404", var"##403", var"##402", var"##401", var"##400", var"##399"; dims = 3)
        var"##406" = (layers[87])(var"##405")
        var"##407" = (layers[88])(var"##406")
        var"##408" = (layers[89])(var"##407")
        var"##409" = (layers[90])(var"##408")
        var"##410" = (layers[91])(var"##409")
    end
end

function _generate_chain(layers::Tuple{Vararg{<:Any,N}}, nodes::Tuple{Vararg{<:Any,N}}, x) where {N}
    # println(m)
    symbols = vcat(:x, [gensym() for _ in 1:N])
    froms = [n.op[2] for n in nodes]
    # symbols = [:x]
    aux_symbols = []
    for i in 1:N
        if length(froms[i]) > 1
            push!(aux_symbols, [gensym() for _ in 1:length(froms[i])]...)
        end
    end
    
    eval(:(symbols = $symbols))
    eval(:(aux_symbols = $aux_symbols))

    # calls_from = [:($(froms[i]) = ((nodes[$i]).op[2])) for i in 1:N]
    # println(Expr(:block, calls_from...))
    # calls = [:($(symbols[i+1]) = layers[$i]((symbols[$(nodes[i])]))) for i in 1:N]
    # println(nodes[1].op[2])
    # calls = [:($(symbols[i+1]) = length($f) == 1 ? layers[$i]($(symbols[(f)])) : layers[$i](($symbols[$(g for g in f)]))) for (i, f) in zip(1:N, froms)]
    # calls = [:($(symbols[i+1]) = length($(froms[i])) == 1 ? layers[$i]($(symbols[(froms[i])])) : layers[$i]($(symbols[froms[i]]))) for i in 1:N]
    calls = []
    k = 1
    for i in 1:N
        if length(froms[i]) == 1
            q = :($(symbols[i+1]) = layers[$i]($(symbols[(froms[i])])))
            push!(calls, q)
        else
            M = length(froms[i])
            # q = :($(aux_symbols[k]) = $(symbols[(froms[i][1])]))
            # q = :($(aux_symbols[k]) = cat($([symbols[(froms[i][j])] for j in 1:M]...); dims=3) )
            q = :($(symbols[i+1]) = cat($([symbols[(froms[i][j])] for j in 1:M]...); dims=3) )
            push!(calls, q)
            k += 1
            # for j in 2:M
            #     :(k = 1)
            #     q = :($(aux_symbols[k]) = cat(($(aux_symbols[k-1]), $(symbols[(froms[i][j])]))...; dims=3))
            #     push!(calls, q)
            #     println(q)
            #     k += 1
            # end
            # q = :($(aux_symbols[k]) = cat(($(aux_symbols[k-1]), $(symbols[(froms[i][j])]))...; dims=3))
            # q = :($(symbols[i+1]) = $(aux_symbols[k-1]))
            # push!(calls, q)
        end
    end
    
    # println(calls)
    # println(symbols,"\n",aux_symbols,"\n",calls)
    eval(:(x = $x))
    eval(:(layers = $layers))
    
    # for e in calls
    #     # println(e)
    #     # eval(e)
    # end
    Expr(:block, calls...)
    # return x
end

# function _apply_chain(layers::Tuple{Vararg{<:Any,N}}, nodes::Tuple{Vararg{<:Any,N}}, x) where {N}
#     # println(m)
#     symbols = vcat(:x, [gensym() for _ in 1:N])
#     froms = [:((nodes[$i].op[2])) for i in 1:N]
#     println("after froms")
#     # symbols = [:x]
#     aux_symbols = []
#     for i in 1:N
#         if length(froms[i]) > 1
#             push!(aux_symbols, [gensym() for _ in 1:length(froms[i])]...)
#         end
#     end
    
#     eval(:(symbols = $symbols))
#     eval(:(aux_symbols = $aux_symbols))

#     # calls_from = [:($(froms[i]) = ((nodes[$i]).op[2])) for i in 1:N]
#     # println(Expr(:block, calls_from...))
#     # calls = [:($(symbols[i+1]) = layers[$i]((symbols[$(nodes[i])]))) for i in 1:N]
#     # println(nodes[1].op[2])
#     # calls = [:($(symbols[i+1]) = length($f) == 1 ? layers[$i]($(symbols[(f)])) : layers[$i](($symbols[$(g for g in f)]))) for (i, f) in zip(1:N, froms)]
#     # calls = [:($(symbols[i+1]) = length($(froms[i])) == 1 ? layers[$i]($(symbols[(froms[i])])) : layers[$i]($(symbols[froms[i]]))) for i in 1:N]
#     calls = []
#     k = 1
#     for i in 1:N
#         if length(froms[i]) == 1
#             q = :($(symbols[i+1]) = layers[$i]($(symbols[(froms[i])])))
#             push!(calls, q)
#         else
#             M = length(froms[i])
#             q = :($(aux_symbols[k]) = $(symbols[(froms[i][1])]))
#             push!(calls, q)
#             k += 1
#             for j in 2:M
#                 :(k = 1)
#                 q = :($(aux_symbols[k]) = cat(($(aux_symbols[k-1]), $(symbols[(froms[i][j])]))...; dims=3))
#                 push!(calls, q)
#                 k += 1
#             end
#             q = :($(symbols[i+1]) = $(aux_symbols[k-1]))
#             push!(calls, q)
#         end
#     end
    
#     # println(calls)
#     # println(symbols,"\n",aux_symbols,"\n",calls)
#     eval(:(x = $x))
#     eval(:(layers = $layers))
    
#     for e in calls
#         # println(e)
#         # eval(e)
#     end
#     Expr(:block, calls...)
#     # return x
# end

function _applychain(m::YOLOChain, x::AbstractArray)
    # fs = m.nodes[1]
    # cs = m.components[1]
    # push!(results, cs(x))
    return _chain(m, m.nodes[end], m.components[end], x)

    # for (i, (node, component)) in enumerate(zip(m.nodes, m.components))
    #     if length(node.parents) == 0
    #         m.results[i] = x
    #     else
    #         if length(node.parents) != 1 || node.parents[1] != -1
    #             x = m.results[node.op[2]]
    #         end
    #         # println(node)
    #         # println(component)
    #         # println(node.op[2])
    #         # println(size(results[node.op[2][1]]))
    #         x = component(x...)
    #         m.results[i] = x
    #     end
    #     # println(length(results))
    # end

    # return x
end

struct Conv
    conv::Flux.Conv
    bn::Flux.BatchNorm
    act::Function
end

Conv(conv::Flux.Conv, bn::Flux.BatchNorm) = Conv(conv, bn, silu)

function Conv(c::Pair{Int64, Int64}, kernel, stride)
    return Conv(
        Flux.Conv((kernel, kernel), c; stride=stride, pad=Flux.SamePad()),
        Flux.BatchNorm(c.second)
    )
end

Flux.@functor Conv

function (m::Conv)(x::AbstractArray)
    # println(x)
    m.act(m.bn(m.conv(x)))
end

Base.show(io::IO, obj::Conv) = show(io, "$(obj.conv); $(obj.bn); $(obj.act)")

# function (m::Conv)(x::Symbol)
#     # println(x)
#     return x
# end

struct SPPCSPC
    cv1::Conv
    cv2::Conv
    cv3::Conv
    cv4::Conv
    m::Array{Flux.MaxPool}
    cv5::Conv
    cv6::Conv
    cv7::Conv
end

# CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
function SPPCSPC(c1, c2; n=1, shortcut=false, g=1, e=0.5, k=(5, 9, 13))
        c_ = Int(2 * c2 * e)  # hidden channels
        cv1 = Conv(Flux.Conv((1, 1), c1 => c_; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))
        cv2 = Conv(Flux.Conv((1, 1), c1 => c_; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))
        cv3 = Conv(Flux.Conv((3, 3), c_ => c_; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))
        cv4 = Conv(Flux.Conv((1, 1), c_ => c_; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))
        m = [Flux.MaxPool((x, x); stride=1, pad=x÷2) for x in k]
        cv5 = Conv(Flux.Conv((1, 1), 4 * c_ => c_; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))
        cv6 = Conv(Flux.Conv((3, 3), c_ => c_; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))
        cv7 = Conv(Flux.Conv((1, 1), 2 * c_ => c2; stride=1, pad=Flux.SamePad()), Flux.BatchNorm(c_))

        return SPPCSPC(cv1, cv2, cv3, cv4, m, cv5, cv6, cv7)
end

function (m::SPPCSPC)(x::AbstractArray)
    x1 = m.cv4(m.cv3(m.cv1(x)))
    c1 = cat(x1, [m(x1) for m in m.m]...; dims=3)
    y1 = m.cv6(m.cv5(c1))
    y2 = m.cv2(x)
    return m.cv7(cat(y1, y2; dims=3))
end

Flux.@functor SPPCSPC