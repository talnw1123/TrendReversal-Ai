import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:fl_chart/fl_chart.dart';
import 'package:intl/intl.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Quant Dashboard',
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0b0e14),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xCC0b0e14),
          elevation: 0,
        ),
      ),
      home: const DashboardScreen(),
    );
  }
}

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  String selectedMarket = 'BTC';
  final List<String> markets = ['BTC', 'US', 'UK', 'Thai', 'Gold'];
  bool isLoading = true;
  
  Map<String, dynamic>? stats;
  List<dynamic> history = [];

  @override
  void initState() {
    super.initState();
    fetchMarketData(selectedMarket);
  }

  Future<void> fetchMarketData(String market) async {
    setState(() {
      isLoading = true;
      selectedMarket = market;
    });

    try {
      final response = await http.get(Uri.parse('http://localhost:8000/api/data?market=$market'));
      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          stats = data['stats'];
          history = data['history'];
          isLoading = false;
        });
      } else {
        throw Exception('Failed to load data');
      }
    } catch (e) {
      setState(() {
        isLoading = false;
      });
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'ANTIGRAVITY TRADING AI',
          style: TextStyle(
            fontWeight: FontWeight.bold,
            letterSpacing: 1.2,
            color: Colors.white,
          ),
        ),
        actions: markets.map<Widget>((market) {
          final isSelected = market == selectedMarket;
          return Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4.0, vertical: 10),
            child: TextButton(
              onPressed: () => fetchMarketData(market),
              style: TextButton.styleFrom(
                backgroundColor: isSelected ? const Color(0x2600d2ff) : Colors.transparent,
                side: BorderSide(
                  color: isSelected ? const Color(0xFF00d2ff) : Colors.white24,
                ),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(20),
                ),
              ),
              child: Text(
                market,
                style: TextStyle(
                  color: isSelected ? const Color(0xFF00d2ff) : Colors.white54,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          );
        }).toList()
        ..add(const SizedBox(width: 16)),
      ),
      body: isLoading
          ? const Center(child: CircularProgressIndicator(color: Color(0xFF00d2ff)))
          : Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _buildStatsRow(),
                  const SizedBox(height: 16),
                  Expanded(
                    child: Container(
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(16),
                      ),
                      padding: const EdgeInsets.all(16.0),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          Text(
                            '$selectedMarket Combined Model Backtest',
                            style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Colors.black),
                            textAlign: TextAlign.center,
                          ),
                          const SizedBox(height: 16),
                          Expanded(child: _buildChart()),
                        ],
                      ),
                    ),
                  ),
                ],
              ),
            ),
    );
  }

  Widget _buildStatsRow() {
    if (stats == null) return const SizedBox.shrink();
    
    int buyCount = 0;
    int sellCount = 0;
    int prevPos = 0;
    for (int i = 0; i < history.length - 1; i++) {
        int currPos = (history[i]['position'] as num).toInt();
        
        if (currPos != prevPos) {
            if (currPos == 1) buyCount++;
            else if (currPos == -1 || currPos == 0) sellCount++;
        }
        prevPos = currPos;
    }

    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        _buildStatCard('RETURN', Text('${stats!['base_return_pct']?.toStringAsFixed(2)}%', style: const TextStyle(color: Color(0xFF00ff88), fontSize: 24, fontWeight: FontWeight.bold))),
        _buildStatCard('B&H RETURN', Text('${stats!['bnh_return_pct']?.toStringAsFixed(2)}%', style: const TextStyle(color: Colors.grey, fontSize: 24, fontWeight: FontWeight.bold))),
        _buildStatCard('MAX DRAWDOWN', Text('${stats!['max_drawdown_pct']?.toStringAsFixed(2)}%', style: const TextStyle(color: Color(0xFFff3366), fontSize: 24, fontWeight: FontWeight.bold))),
        
        Expanded(
          child: Container(
            margin: const EdgeInsets.symmetric(horizontal: 8),
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: const Color(0x99141822),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: Colors.white10),
            ),
            child: Column(
              children: [
                const Text('CHART SIGNALS', style: TextStyle(color: Colors.grey, fontSize: 12, letterSpacing: 1.2)),
                const SizedBox(height: 8),
                FittedBox(
                  fit: BoxFit.scaleDown,
                  child: Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text('$buyCountขึ้น', style: const TextStyle(color: Colors.green, fontSize: 18, fontWeight: FontWeight.bold)),
                      const Text(' / ', style: TextStyle(color: Colors.white54, fontSize: 18)),
                      Text('$sellCountลง', style: const TextStyle(color: Colors.red, fontSize: 18, fontWeight: FontWeight.bold)),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildStatCard(String title, Widget valueWidget) {
    return Expanded(
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: 8),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: const Color(0x99141822),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(color: Colors.white10),
        ),
        child: Column(
          children: [
            Text(title, style: const TextStyle(color: Colors.grey, fontSize: 12, letterSpacing: 1.2)),
            const SizedBox(height: 8),
            valueWidget,
          ],
        ),
      ),
    );
  }

  Widget _buildChart() {
    if (history.isEmpty) return const Center(child: Text('No data available', style: TextStyle(color: Colors.black)));

    List<FlSpot> priceSpots = [];
    List<FlSpot> equitySpots = [];
    List<FlSpot> bnhSpots = [];
    
    List<ScatterSpot> buySpots = [];
    List<ScatterSpot> sellSpots = [];
    List<VerticalRangeAnnotation> trendAnnotations = [];

    double minPrice = double.infinity;
    double maxPrice = double.negativeInfinity;
    
    double minEq = double.infinity;
    double maxEq = double.negativeInfinity;

    int prevPos = 0;
    
    // For shaded regions
    int startIdx = 0;
    int currentTrend = -1; // 1 = Uptrend, 0 = Downtrend
    
    for (int i = 0; i < history.length; i++) {
        final row = history[i];
        final price = row['price'] is num ? (row['price'] as num).toDouble() : double.parse(row['price'].toString());
        final eq = row['equity_curve'] is num ? (row['equity_curve'] as num).toDouble() : double.parse(row['equity_curve'].toString());
        final bnh = row['bnh_curve'] is num ? (row['bnh_curve'] as num).toDouble() : double.parse(row['bnh_curve'].toString());
        
        final trendStr = row['trend_regime'].toString();
        int rowTrend = trendStr.contains("1") ? 1 : 0;
        
        if (i == 0) {
          currentTrend = rowTrend;
          startIdx = i;
        } else if (rowTrend != currentTrend) {
          // Trend changed, close the previous region
          trendAnnotations.add(
            VerticalRangeAnnotation(
              x1: startIdx.toDouble(),
              x2: i.toDouble(),
              color: currentTrend == 1 ? Colors.green.withOpacity(0.1) : Colors.red.withOpacity(0.1),
            )
          );
          startIdx = i;
          currentTrend = rowTrend;
        }
        
        double x = i.toDouble();
        priceSpots.add(FlSpot(x, price));
        equitySpots.add(FlSpot(x, eq));
        bnhSpots.add(FlSpot(x, bnh));

        if (price < minPrice) minPrice = price;
        if (price > maxPrice) maxPrice = price;
        if (eq < minEq) minEq = eq;
        if (bnh < minEq) minEq = bnh;
        if (eq > maxEq) maxEq = eq;
        if (bnh > maxEq) maxEq = bnh;

        // Plotting markers shifted by 1 day to match Python plot_backtest
        if (i < history.length - 1) {
            int currPos = (row['position'] as num).toInt();
            
            if (currPos != prevPos) {
                final nextRow = history[i + 1];
                final nextPrice = nextRow['price'] is num ? (nextRow['price'] as num).toDouble() : double.parse(nextRow['price'].toString());
                
                if (currPos == 1) {
                    buySpots.add(ScatterSpot((i + 1).toDouble(), nextPrice));
                } else if (currPos == -1 || currPos == 0) {
                    sellSpots.add(ScatterSpot((i + 1).toDouble(), nextPrice));
                }
            }
            prevPos = currPos;
        }
    }

    // Add the final trend region
    if (history.isNotEmpty) {
      trendAnnotations.add(
        VerticalRangeAnnotation(
          x1: startIdx.toDouble(),
          x2: (history.length - 1).toDouble(),
          color: currentTrend == 1 ? Colors.green.withOpacity(0.1) : Colors.red.withOpacity(0.1),
        )
      );
    }

    // Add padding to chart scales
    minPrice = minPrice * 0.95;
    maxPrice = maxPrice * 1.05;
    minEq = minEq * 0.95;
    maxEq = maxEq * 1.05;

    return Column(
      children: [
        // Legend Row
        Wrap(
          spacing: 16,
          runSpacing: 8,
          alignment: WrapAlignment.center,
          children: [
            _buildLegendItem(Colors.black, 'Price', isLine: true),
            _buildLegendItem(Colors.green, 'Buy/Long', isTriangleUp: true),
            _buildLegendItem(Colors.red, 'Sell/Exit', isTriangleDown: true),
            _buildLegendItem(Colors.blue, 'Strategy', isLine: true, lineWidth: 2),
            _buildLegendItem(Colors.grey, 'Buy & Hold', isLine: true, isDashed: true),
          ],
        ),
        const SizedBox(height: 16),
        const SizedBox(
          width: double.infinity,
          child: Text("Price & Signals", style: TextStyle(color: Colors.black87, fontWeight: FontWeight.bold, fontSize: 12), textAlign: TextAlign.left),
        ),
        Expanded(
          flex: 1,
          child: LineChart(
            LineChartData(
              rangeAnnotations: RangeAnnotations(
                verticalRangeAnnotations: trendAnnotations,
              ),
              lineTouchData: const LineTouchData(enabled: false), // Basic interaction
              gridData: const FlGridData(show: true, drawVerticalLine: false),
              titlesData: FlTitlesData(
                show: true,
                rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                bottomTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)), // Hide x-axis labels on top chart
                leftTitles: AxisTitles(
                  axisNameWidget: const Text("Price", style: TextStyle(color: Colors.black54, fontSize: 10)),
                  sideTitles: SideTitles(showTitles: true, reservedSize: 45, getTitlesWidget: (value, meta) => Text(value.toStringAsFixed(0), style: const TextStyle(color: Colors.black54, fontSize: 10))),
                ),
              ),
              borderData: FlBorderData(show: true, border: Border.all(color: Colors.black12)),
              minX: 0,
              maxX: (history.length - 1).toDouble(),
              minY: minPrice,
              maxY: maxPrice,
              lineBarsData: [
                LineChartBarData(
                  spots: priceSpots,
                  isCurved: false,
                  color: Colors.black,
                  barWidth: 1.5,
                  isStrokeCapRound: true,
                  dotData: FlDotData(
                    show: true,
                    checkToShowDot: (spot, barData) {
                        return buySpots.any((s) => s.x == spot.x) || sellSpots.any((s) => s.x == spot.x);
                    },
                    getDotPainter: (spot, percent, barData, index) {
                        if (buySpots.any((s) => s.x == spot.x)) {
                            return FlDotTrianglePainter(size: 14, color: Colors.green, isDown: false);
                        }
                        if (sellSpots.any((s) => s.x == spot.x)) {
                            return FlDotTrianglePainter(size: 14, color: Colors.red, isDown: true);
                        }
                        return FlDotCirclePainter(radius: 0);
                    }
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 20),
        const SizedBox(
          width: double.infinity,
          child: Text("Equity Curve", style: TextStyle(color: Colors.black87, fontWeight: FontWeight.bold, fontSize: 12), textAlign: TextAlign.left),
        ),
        Expanded(
          flex: 1,
          child: LineChart(
            LineChartData(
              rangeAnnotations: RangeAnnotations(
                verticalRangeAnnotations: trendAnnotations,
              ),
              lineTouchData: const LineTouchData(enabled: true),
              gridData: const FlGridData(show: true, drawVerticalLine: false),
              titlesData: FlTitlesData(
                show: true,
                rightTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                topTitles: const AxisTitles(sideTitles: SideTitles(showTitles: false)),
                bottomTitles: AxisTitles(
                  sideTitles: SideTitles(
                    showTitles: true,
                    reservedSize: 30,
                    getTitlesWidget: (value, meta) {
                      if (value.toInt() >= 0 && value.toInt() < history.length && value.toInt() % (history.length ~/ 6) == 0) {
                        final dateStr = history[value.toInt()]['date'];
                        final date = DateTime.parse(dateStr);
                        return Padding(padding: const EdgeInsets.only(top: 8.0), child: Text(DateFormat('MMM yyyy').format(date), style: const TextStyle(color: Colors.black54, fontSize: 10)));
                      }
                      return const Text('');
                    }
                  ),
                ),
                leftTitles: AxisTitles(
                  axisNameWidget: const Text("Equity (\$)", style: TextStyle(color: Colors.black54, fontSize: 10)),
                  sideTitles: SideTitles(showTitles: true, reservedSize: 45, getTitlesWidget: (value, meta) => Text(value.toStringAsFixed(0), style: const TextStyle(color: Colors.black54, fontSize: 10))),
                ),
              ),
              borderData: FlBorderData(show: true, border: Border.all(color: Colors.black12)),
              minX: 0,
              maxX: (history.length - 1).toDouble(),
              minY: minEq,
              maxY: maxEq,
              lineBarsData: [
                LineChartBarData(
                  spots: equitySpots,
                  isCurved: false,
                  color: Colors.blue,
                  barWidth: 2,
                  dotData: const FlDotData(show: false),
                ),
                LineChartBarData(
                  spots: bnhSpots,
                  isCurved: false,
                  color: Colors.grey,
                  barWidth: 1.5,
                  dashArray: [5, 5],
                  dotData: const FlDotData(show: false),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildLegendItem(Color color, String text, {bool isLine = false, bool isDashed = false, bool isTriangleUp = false, bool isTriangleDown = false, double lineWidth = 1.5}) {
    Widget icon;
    if (isLine) {
      icon = Row(
        mainAxisSize: MainAxisSize.min,
        children: isDashed 
            ? [
                Container(width: 4, height: lineWidth, color: color),
                const SizedBox(width: 2),
                Container(width: 4, height: lineWidth, color: color),
                const SizedBox(width: 2),
                Container(width: 4, height: lineWidth, color: color),
              ]
            : [Container(width: 16, height: lineWidth, color: color)],
      );
    } else if (isTriangleUp) {
      icon = Icon(Icons.arrow_drop_up, color: color, size: 24);
    } else if (isTriangleDown) {
      icon = Icon(Icons.arrow_drop_down, color: color, size: 24);
    } else {
      icon = Container(width: 10, height: 10, color: color);
    }

    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        icon,
        const SizedBox(width: 4),
        Text(text, style: const TextStyle(color: Colors.black87, fontSize: 10, fontWeight: FontWeight.bold)),
      ],
    );
  }
}

class FlDotTrianglePainter extends FlDotPainter {
  final double size;
  final Color color;
  final bool isDown;

  FlDotTrianglePainter({required this.size, required this.color, this.isDown = false});

  @override
  Color get mainColor => color;

  @override
  FlDotPainter lerp(FlDotPainter a, FlDotPainter b, double t) {
    return this; // No animation interpolation needed
  }

  @override
  void draw(Canvas canvas, FlSpot spot, Offset offsetInCanvas) {
    if (isDown) {
      final path = Path()
        ..moveTo(offsetInCanvas.dx, offsetInCanvas.dy + size / 2)
        ..lineTo(offsetInCanvas.dx - size / 2, offsetInCanvas.dy - size / 2)
        ..lineTo(offsetInCanvas.dx + size / 2, offsetInCanvas.dy - size / 2)
        ..close();
      canvas.drawPath(path, Paint()..color = color);
    } else {
      final path = Path()
        ..moveTo(offsetInCanvas.dx, offsetInCanvas.dy - size / 2)
        ..lineTo(offsetInCanvas.dx - size / 2, offsetInCanvas.dy + size / 2)
        ..lineTo(offsetInCanvas.dx + size / 2, offsetInCanvas.dy + size / 2)
        ..close();
      canvas.drawPath(path, Paint()..color = color);
    }
  }

  @override
  Size getSize(FlSpot spot) => Size(size, size);

  @override
  List<Object?> get props => [size, color, isDown];
}
