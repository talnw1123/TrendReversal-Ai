import 'dart:convert';
import 'dart:html' as html;
import 'dart:js' as js;
import 'dart:ui_web' as ui;
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Quant Dashboard',
      debugShowCheckedModeBanner: false,
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

// ─── Market labels for display ───
const Map<String, String> marketSymbols = {
  'BTC': 'BITSTAMP:BTCUSD',
  'US': 'TVC:SPX',
  'UK': 'CAPITALCOM:UK100',
  'Thai': 'SET:SET',
  'Gold': 'OANDA:XAUUSD',
};

// ─── Price + Signals Chart (Lightweight Charts) ───
class PriceSignalsChartWidget extends StatefulWidget {
  final List<dynamic> history;
  final String market;
  final bool isDarkTheme;

  const PriceSignalsChartWidget({
    super.key,
    required this.history,
    required this.market,
    required this.isDarkTheme,
  });

  @override
  State<PriceSignalsChartWidget> createState() => _PriceSignalsChartWidgetState();
}

class _PriceSignalsChartWidgetState extends State<PriceSignalsChartWidget> {
  late String _viewType;

  @override
  void initState() {
    super.initState();
    _viewType = 'price-${widget.market}-${DateTime.now().millisecondsSinceEpoch}';
    _registerView();
  }

  @override
  void didUpdateWidget(PriceSignalsChartWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.market != widget.market || oldWidget.history.length != widget.history.length || oldWidget.isDarkTheme != widget.isDarkTheme) {
      _viewType = 'price-${widget.market}-${DateTime.now().millisecondsSinceEpoch}';
      _registerView();
      setState(() {});
    }
  }

  void _registerView() {
    final historyJson = json.encode(widget.history);
    final market = widget.market;
    final isDark = widget.isDarkTheme ? 'true' : 'false';

    // ignore: undefined_prefixed_name
    ui.platformViewRegistry.registerViewFactory(_viewType, (int viewId) {
      final container = html.DivElement()
        ..id = 'price_container_${market}_$viewId'
        ..style.width = '100%'
        ..style.height = '100%'
        ..style.position = 'relative';

      Future.delayed(const Duration(milliseconds: 50), () {
        js.context.callMethod('eval', [
          '''
          (function() {
            var container = document.getElementById("${container.id}");
            if (!container || !window.LightweightCharts) return;

            var chart = LightweightCharts.createChart(container, {
              layout: {
                background: { type: 'solid', color: ${isDark == 'true' ? "'#0b0e14'" : "'#ffffff'"} },
                textColor: ${isDark == 'true' ? "'#9B9EA3'" : "'#424242'"},
                fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
              },
              grid: {
                vertLines: { color: ${isDark == 'true' ? "'rgba(255,255,255,0.04)'" : "'rgba(0,0,0,0.04)'"} },
                horzLines: { color: ${isDark == 'true' ? "'rgba(255,255,255,0.04)'" : "'rgba(0,0,0,0.04)'"} },
              },
              crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: { color: 'rgba(0,210,255,0.3)', width: 1, style: 2 },
                horzLine: { color: 'rgba(0,210,255,0.3)', width: 1, style: 2 },
              },
              rightPriceScale: { borderColor: ${isDark == 'true' ? "'rgba(255,255,255,0.1)'" : "'rgba(0,0,0,0.1)'"} },
              timeScale: { borderColor: ${isDark == 'true' ? "'rgba(255,255,255,0.1)'" : "'rgba(0,0,0,0.1)'"}, timeVisible: false },
              handleScroll: true,
              handleScale: true,
            });

            var resizeObserver = new ResizeObserver(function(entries) {
              for (var entry of entries) {
                chart.applyOptions({ width: entry.contentRect.width, height: entry.contentRect.height });
              }
            });
            resizeObserver.observe(container);

            var history = $historyJson;

            // ── Build data arrays ──
            var priceData = [];
            var markers = [];
            var prevPos = 0;

            // Position zone background highlighting
            var zoneBuyData = [];   // green zones (in position)
            var zoneSellData = [];  // red zones (out of position)

            for (var i = 0; i < history.length; i++) {
              var row = history[i];
              var dateStr = row["date"];
              var price = parseFloat(row["price"]) || 0;
              var pos = parseFloat(row["position"]) || 0;

              if (price > 0) {
                priceData.push({ time: dateStr, value: price });
              }

              // Detect position changes for BUY/SELL markers
              if (i > 0 && pos !== prevPos) {
                if (pos > 0) {
                  markers.push({
                    time: dateStr,
                    position: 'belowBar',
                    color: '#26a69a',
                    shape: 'arrowUp',
                    text: 'BUY',
                  });
                } else {
                  markers.push({
                    time: dateStr,
                    position: 'aboveBar',
                    color: '#ef5350',
                    shape: 'arrowDown',
                    text: 'SELL',
                  });
                }
              }
              prevPos = pos;
            }

            // ── Price Line Series (grey like reference) ──
            var priceSeries = chart.addLineSeries({
              color: '#888888',
              lineWidth: 2,
              crosshairMarkerVisible: true,
              crosshairMarkerRadius: 4,
              priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
              title: '$market Price',
              lastValueVisible: true,
              priceLineVisible: true,
            });
            priceSeries.setData(priceData);

            // ── Position zone background (green/red shading) ──
            // We create two area series with very high/low base to simulate background
            // Green = in position, Red/pink = out of position
            if (priceData.length > 0) {
              // Find price range for shading height
              var maxPrice = 0;
              for (var k = 0; k < priceData.length; k++) {
                if (priceData[k].value > maxPrice) maxPrice = priceData[k].value;
              }
              var shadeTop = maxPrice * 1.2;

              // Green background (in-position zones)
              var greenData = [];
              // Red background (out-of-position zones)
              var redData = [];

              for (var j = 0; j < history.length; j++) {
                var row2 = history[j];
                var dt = row2["date"];
                var p = parseFloat(row2["position"]) || 0;

                if (p > 0) {
                  greenData.push({ time: dt, value: shadeTop });
                  redData.push({ time: dt, value: 0 });
                } else {
                  greenData.push({ time: dt, value: 0 });
                  redData.push({ time: dt, value: shadeTop });
                }
              }

              var greenBg = chart.addAreaSeries({
                topColor: 'rgba(38, 166, 154, 0.08)',
                bottomColor: 'rgba(38, 166, 154, 0.01)',
                lineColor: 'rgba(0,0,0,0)',
                lineWidth: 0,
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
              });
              greenBg.setData(greenData);

              var redBg = chart.addAreaSeries({
                topColor: 'rgba(239, 83, 80, 0.06)',
                bottomColor: 'rgba(239, 83, 80, 0.01)',
                lineColor: 'rgba(0,0,0,0)',
                lineWidth: 0,
                priceLineVisible: false,
                lastValueVisible: false,
                crosshairMarkerVisible: false,
              });
              redBg.setData(redData);
            }

            // ── Set BUY/SELL markers ──
            if (markers.length > 0) {
              priceSeries.setMarkers(markers);
            }

            // ── Watermark ──
            chart.applyOptions({
              watermark: {
                visible: true,
                fontSize: 48,
                horzAlign: 'center',
                vertAlign: 'center',
                color: ${isDark == 'true' ? "'rgba(255, 255, 255, 0.04)'" : "'rgba(0, 0, 0, 0.04)'"},
                text: '$market Combined - Price & Signals',
              },
            });

            chart.timeScale().fitContent();
          })();
          '''
        ]);
      });

      return container;
    });
  }

  @override
  Widget build(BuildContext context) {
    return HtmlElementView(key: ValueKey(_viewType), viewType: _viewType);
  }
}

// ─── Equity Curve Chart ───
class EquityCurveChartWidget extends StatefulWidget {
  final List<dynamic> history;
  final String market;
  final bool isDarkTheme;
  final double strategyReturn;
  final double bnhReturn;

  const EquityCurveChartWidget({
    super.key,
    required this.history,
    required this.market,
    required this.isDarkTheme,
    required this.strategyReturn,
    required this.bnhReturn,
  });

  @override
  State<EquityCurveChartWidget> createState() => _EquityCurveChartWidgetState();
}

class _EquityCurveChartWidgetState extends State<EquityCurveChartWidget> {
  late String _viewType;

  @override
  void initState() {
    super.initState();
    _viewType = 'eq-${widget.market}-${DateTime.now().millisecondsSinceEpoch}';
    _registerView();
  }

  @override
  void didUpdateWidget(EquityCurveChartWidget oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.market != widget.market || oldWidget.history.length != widget.history.length || oldWidget.isDarkTheme != widget.isDarkTheme) {
      _viewType = 'eq-${widget.market}-${DateTime.now().millisecondsSinceEpoch}';
      _registerView();
      setState(() {});
    }
  }

  void _registerView() {
    final historyJson = json.encode(widget.history);
    final market = widget.market;
    final isDark = widget.isDarkTheme ? 'true' : 'false';
    final stratRet = widget.strategyReturn.toStringAsFixed(2);
    final bnhRet = widget.bnhReturn.toStringAsFixed(2);

    // ignore: undefined_prefixed_name
    ui.platformViewRegistry.registerViewFactory(_viewType, (int viewId) {
      final container = html.DivElement()
        ..id = 'eq_container_${market}_$viewId'
        ..style.width = '100%'
        ..style.height = '100%';

      Future.delayed(const Duration(milliseconds: 80), () {
        js.context.callMethod('eval', [
          '''
          (function() {
            var container = document.getElementById("${container.id}");
            if (!container || !window.LightweightCharts) return;

            var chart = LightweightCharts.createChart(container, {
              layout: {
                background: { type: 'solid', color: ${isDark == 'true' ? "'#0b0e14'" : "'#ffffff'"} },
                textColor: ${isDark == 'true' ? "'#9B9EA3'" : "'#424242'"},
                fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
              },
              grid: {
                vertLines: { color: ${isDark == 'true' ? "'rgba(255,255,255,0.04)'" : "'rgba(0,0,0,0.04)'"} },
                horzLines: { color: ${isDark == 'true' ? "'rgba(255,255,255,0.04)'" : "'rgba(0,0,0,0.04)'"} },
              },
              rightPriceScale: { borderColor: ${isDark == 'true' ? "'rgba(255,255,255,0.1)'" : "'rgba(0,0,0,0.1)'"} },
              timeScale: { borderColor: ${isDark == 'true' ? "'rgba(255,255,255,0.1)'" : "'rgba(0,0,0,0.1)'"}, timeVisible: false },
              handleScroll: true,
              handleScale: true,
              crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: { color: 'rgba(0,210,255,0.3)', width: 1, style: 2 },
                horzLine: { color: 'rgba(0,210,255,0.3)', width: 1, style: 2 },
              },
            });

            var resizeObserver = new ResizeObserver(function(entries) {
              for (var entry of entries) {
                chart.applyOptions({ width: entry.contentRect.width, height: entry.contentRect.height });
              }
            });
            resizeObserver.observe(container);

            var history = $historyJson;
            var equityData = [];
            var bnhData = [];

            for (var i = 0; i < history.length; i++) {
              var row = history[i];
              var dateStr = row["date"];
              var eq = parseFloat(row["equity_curve"]) || 0;
              var bnh = parseFloat(row["bnh_curve"]) || 0;
              if (eq > 0) equityData.push({ time: dateStr, value: eq });
              if (bnh > 0) bnhData.push({ time: dateStr, value: bnh });
            }

            // Strategy equity curve (solid blue line, like reference)
            var eqSeries = chart.addLineSeries({
              color: '#1565C0',
              lineWidth: 2,
              title: 'Strategy ($stratRet%)',
              priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
              lastValueVisible: true,
              priceLineVisible: true,
            });
            eqSeries.setData(equityData);

            // Buy & Hold curve (dashed grey line, like reference)
            var bnhSeries = chart.addLineSeries({
              color: '#888888',
              lineWidth: 1,
              lineStyle: 2,
              title: 'Buy & Hold ($bnhRet%)',
              priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
              lastValueVisible: true,
              priceLineVisible: false,
            });
            bnhSeries.setData(bnhData);

            chart.applyOptions({
              watermark: {
                visible: true,
                fontSize: 32,
                horzAlign: 'center',
                vertAlign: 'center',
                color: ${isDark == 'true' ? "'rgba(255, 255, 255, 0.04)'" : "'rgba(0, 0, 0, 0.04)'"},
                text: 'Equity Curve (Hybrid_MOO)',
              },
            });

            chart.timeScale().fitContent();
          })();
          '''
        ]);
      });

      return container;
    });
  }

  @override
  Widget build(BuildContext context) {
    return HtmlElementView(key: ValueKey(_viewType), viewType: _viewType);
  }
}

// ─── Dashboard Screen ───
class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  bool isDarkTheme = true;

  Color get bgCol => isDarkTheme ? const Color(0xFF0b0e14) : const Color(0xFFF4F6F8);
  Color get panelCol => isDarkTheme ? const Color(0xFF111620) : Colors.white;
  Color get borderCol => isDarkTheme ? Colors.white10 : Colors.black12;
  Color get textPri => isDarkTheme ? Colors.white : Colors.black87;
  Color get textSec => isDarkTheme ? Colors.white70 : Colors.black54;
  Color get textMut => isDarkTheme ? Colors.white38 : Colors.black38;
  Color get textFai => isDarkTheme ? Colors.white24 : Colors.black26;
  Color get cardCol => isDarkTheme ? const Color(0x99141822) : Colors.white;
  Color get appBarCol => isDarkTheme ? const Color(0xCC0b0e14) : Colors.white;

  String selectedMarket = 'BTC';
  final List<String> markets = ['BTC', 'US', 'UK', 'Thai', 'Gold'];
  bool isLoading = true;
  bool apiAvailable = true;

  Map<String, dynamic>? stats;
  Map<String, dynamic>? currentSignal;
  List<dynamic> signalMarkers = [];
  List<dynamic> history = [];

  @override
  void initState() {
    super.initState();
    fetchData(selectedMarket);
  }

  Future<void> fetchData(String market) async {
    setState(() {
      isLoading = true;
      selectedMarket = market;
    });

    try {
      final response = await http.get(
        Uri.parse('http://localhost:5000/api/data?market=$market'),
      );

      Map<String, dynamic>? newStats;
      List<dynamic> newHistory = [];
      Map<String, dynamic>? newCurrentSignal;
      List<dynamic> newMarkers = [];

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        newStats = data['stats'];
        newHistory = data['history'] ?? [];

        // Derive signal markers from history (position changes = BUY/SELL)
        double prevPosition = 0;
        for (int i = 0; i < newHistory.length; i++) {
          final row = newHistory[i];
          final pos = (row['position'] as num?)?.toDouble() ?? 0;
          if (i > 0 && pos != prevPosition) {
            newMarkers.add({
              'type': pos > 0 ? 'BUY' : 'SELL',
              'date': row['date'],
              'price': row['price'],
              'ml_up_prob': row['ml_up_prob'],
              'ml_down_prob': row['ml_down_prob'],
              'trend': row['trend_regime'],
            });
          }
          prevPosition = pos;
        }

        // Keep only newest 20 markers
        if (newMarkers.length > 20) {
          newMarkers = newMarkers.sublist(newMarkers.length - 20);
        }

        // Current signal = last row in history
        if (newHistory.isNotEmpty) {
          final last = newHistory.last;
          newCurrentSignal = {
            'action': last['signal_action'] ?? 'WAIT',
            'price': last['price'],
            'date': last['date'],
            'ml_up_prob': last['ml_up_prob'],
            'ml_down_prob': last['ml_down_prob'],
            'trend': last['trend_regime'],
          };
        }
      }

      setState(() {
        stats = newStats;
        history = newHistory;
        currentSignal = newCurrentSignal;
        signalMarkers = newMarkers;
        isLoading = false;
        apiAvailable = true;
      });
    } catch (e) {
      setState(() {
        isLoading = false;
        apiAvailable = false;
      });
    }
  }

  double get strategyReturn => (stats?['base_return_pct'] as num?)?.toDouble() ?? 0.0;
  double get bnhReturn => (stats?['bnh_return_pct'] as num?)?.toDouble() ?? 0.0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: bgCol,
      appBar: AppBar(
        backgroundColor: appBarCol,
        title: Text(
          '⚡ ANTIGRAVITY TRADING AI',
          style: TextStyle(fontWeight: FontWeight.bold, letterSpacing: 1.2, color: textPri, fontSize: 16),
          overflow: TextOverflow.ellipsis,
        ),
        actions: [
          IconButton(
            icon: Icon(isDarkTheme ? Icons.light_mode : Icons.dark_mode, color: textPri),
            onPressed: () {
              setState(() {
                isDarkTheme = !isDarkTheme;
              });
            },
          ),
          ...markets.map<Widget>((market) {
          final isSelected = market == selectedMarket;
          return Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4.0, vertical: 10),
            child: TextButton(
              onPressed: () => fetchData(market),
              style: TextButton.styleFrom(
                backgroundColor: isSelected ? Color(0xFF00d2ff).withOpacity(0.15) : Colors.transparent,
                side: BorderSide(
                  color: isSelected ? Color(0xFF00d2ff) : textFai,
                  width: isSelected ? 1.5 : 1,
                ),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                padding: const EdgeInsets.symmetric(horizontal: 16),
              ),
              child: Text(
                market,
                style: TextStyle(
                  color: isSelected ? Color(0xFF00d2ff) : textSec,
                  fontWeight: FontWeight.bold,
                  fontSize: 13,
                ),
              ),
            ),
          );
        }).toList(),
          SizedBox(width: 16),
        ],
      ),
      body: isLoading
          ? Center(child: CircularProgressIndicator(color: Color(0xFF00d2ff)))
          : Padding(
              padding: const EdgeInsets.all(12.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  _buildStatsRow(),
                  SizedBox(height: 10),
                  Expanded(
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        // Left: Charts
                        Expanded(
                          flex: 3,
                          child: Column(
                            children: [
                              // Price + Signals Chart
                              Expanded(
                                flex: 3,
                                child: Container(
                                  decoration: BoxDecoration(
                                    borderRadius: BorderRadius.circular(12),
                                    border: Border.all(color: borderCol),
                                  ),
                                  clipBehavior: Clip.antiAlias,
                                  child: history.isNotEmpty
                                      ? PriceSignalsChartWidget(history: history, market: selectedMarket, isDarkTheme: isDarkTheme)
                                      : Center(child: Text('No price data', style: TextStyle(color: textFai))),
                                ),
                              ),
                              SizedBox(height: 8),
                              // Equity Curve
                              Expanded(
                                flex: 2,
                                child: Container(
                                  decoration: BoxDecoration(
                                    borderRadius: BorderRadius.circular(12),
                                    border: Border.all(color: borderCol),
                                  ),
                                  clipBehavior: Clip.antiAlias,
                                  child: history.isNotEmpty
                                      ? EquityCurveChartWidget(
                                          history: history,
                                          market: selectedMarket,
                                          strategyReturn: strategyReturn,
                                          bnhReturn: bnhReturn,
                                          isDarkTheme: isDarkTheme,
                                        )
                                      : Center(child: Text('No equity data', style: TextStyle(color: textFai))),
                                ),
                              ),
                            ],
                          ),
                        ),
                        SizedBox(width: 12),
                        // Right: Signal Panel
                        SizedBox(width: 280, child: _buildSignalPanel()),
                      ],
                    ),
                  ),
                ],
              ),
            ),
    );
  }

  // ─── Signal Panel ───
  Widget _buildSignalPanel() {
    return Container(
      decoration: BoxDecoration(
        color: panelCol,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: borderCol),
      ),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            decoration: BoxDecoration(border: Border(bottom: BorderSide(color: borderCol))),
            child: Row(
              children: [
                Container(
                  width: 8, height: 8,
                  decoration: BoxDecoration(
                    color: apiAvailable ? Color(0xFF26a69a) : Colors.orange,
                    shape: BoxShape.circle,
                    boxShadow: [BoxShadow(color: (apiAvailable ? Color(0xFF26a69a) : Colors.orange).withOpacity(0.5), blurRadius: 6)],
                  ),
                ),
                SizedBox(width: 10),
                Text('AI SIGNAL FEED', style: TextStyle(color: textSec, fontSize: 12, fontWeight: FontWeight.bold, letterSpacing: 1.2)),
                const Spacer(),
                Text(selectedMarket, style: TextStyle(color: Color(0xFF00d2ff), fontSize: 12, fontWeight: FontWeight.bold)),
              ],
            ),
          ),
          _buildCurrentSignalBanner(),
          Expanded(child: _buildSignalHistory()),
        ],
      ),
    );
  }

  Widget _buildCurrentSignalBanner() {
    if (!apiAvailable || currentSignal == null) {
      return Container(
        margin: const EdgeInsets.all(12),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(color: borderCol, borderRadius: BorderRadius.circular(10), border: Border.all(color: borderCol)),
        child: Center(child: Text('No signal data', style: TextStyle(color: textFai, fontSize: 12))),
      );
    }

    final action = currentSignal!['action']?.toString() ?? 'WAIT';
    final price = currentSignal!['price'];
    final date = currentSignal!['date']?.toString() ?? '';
    final mlUp = currentSignal!['ml_up_prob'];
    final mlDown = currentSignal!['ml_down_prob'];
    final trend = currentSignal!['trend']?.toString() ?? '';

    Color actionColor;
    IconData actionIcon;
    String actionLabel;

    switch (action) {
      case 'BUY':
        actionColor = Color(0xFF26a69a);
        actionIcon = Icons.arrow_upward_rounded;
        actionLabel = '🟢 BUY';
        break;
      case 'SELL':
        actionColor = Color(0xFFef5350);
        actionIcon = Icons.arrow_downward_rounded;
        actionLabel = '🔴 SELL';
        break;
      case 'HOLD':
        actionColor = Color(0xFF1565C0);
        actionIcon = Icons.pause_circle_outline;
        actionLabel = '🔵 HOLD';
        break;
      default:
        actionColor = Colors.grey;
        actionIcon = Icons.hourglass_empty;
        actionLabel = '⏳ WAIT';
    }

    final isUptrend = trend.contains('1');

    return Container(
      margin: const EdgeInsets.all(12),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        gradient: LinearGradient(begin: Alignment.topLeft, end: Alignment.bottomRight, colors: [actionColor.withOpacity(0.12), actionColor.withOpacity(0.04)]),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: actionColor.withOpacity(0.3)),
      ),
      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
        Row(children: [
          Icon(actionIcon, color: actionColor, size: 22),
          SizedBox(width: 8),
          Text(actionLabel, style: TextStyle(color: actionColor, fontSize: 20, fontWeight: FontWeight.bold)),
        ]),
        SizedBox(height: 10),
        Row(children: [
          Text('Price: ', style: TextStyle(color: textMut, fontSize: 11)),
          Text(price is num ? price.toStringAsFixed(2) : price.toString(), style: TextStyle(color: textPri, fontSize: 13, fontWeight: FontWeight.bold)),
        ]),
        SizedBox(height: 4),
        Row(children: [
          Text('Trend: ', style: TextStyle(color: textMut, fontSize: 11)),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
            decoration: BoxDecoration(color: (isUptrend ? Color(0xFF26a69a) : Color(0xFFef5350)).withOpacity(0.15), borderRadius: BorderRadius.circular(4)),
            child: Text(isUptrend ? '▲ Uptrend' : '▼ Downtrend', style: TextStyle(color: isUptrend ? Color(0xFF26a69a) : Color(0xFFef5350), fontSize: 11, fontWeight: FontWeight.bold)),
          ),
        ]),
        SizedBox(height: 4),
        Row(children: [
          Text('ML Up: ', style: TextStyle(color: textMut, fontSize: 11)),
          Text(mlUp is num ? '${mlUp.toStringAsFixed(1)}%' : '$mlUp%', style: TextStyle(color: Color(0xFF26a69a), fontSize: 12, fontWeight: FontWeight.bold)),
          SizedBox(width: 12),
          Text('Down: ', style: TextStyle(color: textMut, fontSize: 11)),
          Text(mlDown is num ? '${mlDown.toStringAsFixed(1)}%' : '$mlDown%', style: TextStyle(color: Color(0xFFef5350), fontSize: 12, fontWeight: FontWeight.bold)),
        ]),
        SizedBox(height: 6),
        Text(date, style: TextStyle(color: textFai, fontSize: 10)),
      ]),
    );
  }

  Widget _buildSignalHistory() {
    if (!apiAvailable || signalMarkers.isEmpty) {
      return Center(child: Text('No signal history', style: TextStyle(color: textFai, fontSize: 12)));
    }

    final reversed = signalMarkers.reversed.toList();

    return ListView.builder(
      padding: const EdgeInsets.symmetric(horizontal: 12),
      itemCount: reversed.length,
      itemBuilder: (context, index) {
        final marker = reversed[index];
        final isBuy = marker['type'] == 'BUY';
        final color = isBuy ? Color(0xFF26a69a) : Color(0xFFef5350);
        final icon = isBuy ? Icons.arrow_upward_rounded : Icons.arrow_downward_rounded;
        final price = marker['price'];
        final date = marker['date']?.toString() ?? '';
        final mlUp = marker['ml_up_prob'];

        return Container(
          margin: const EdgeInsets.only(bottom: 6),
          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
          decoration: BoxDecoration(color: color.withOpacity(0.06), borderRadius: BorderRadius.circular(8), border: Border.all(color: color.withOpacity(0.15))),
          child: Row(children: [
            Container(
              width: 28, height: 28,
              decoration: BoxDecoration(color: color.withOpacity(0.15), borderRadius: BorderRadius.circular(6)),
              child: Icon(icon, color: color, size: 16),
            ),
            SizedBox(width: 10),
            Expanded(
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Row(children: [
                  Text(marker['type'], style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.bold)),
                  const Spacer(),
                  Text(date, style: TextStyle(color: textFai, fontSize: 10)),
                ]),
                SizedBox(height: 2),
                Row(children: [
                  Text(price is num ? '\$${price.toStringAsFixed(2)}' : '\$$price', style: TextStyle(color: textSec, fontSize: 11)),
                  if (mlUp != null) ...[
                    const Spacer(),
                    Text('ML: ${mlUp is num ? mlUp.toStringAsFixed(1) : mlUp}%', style: TextStyle(color: textFai, fontSize: 10)),
                  ],
                ]),
              ]),
            ),
          ]),
        );
      },
    );
  }

  // ─── Stats Row ───
  Widget _buildStatsRow() {
    if (!apiAvailable) {
      return Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(color: cardCol, borderRadius: BorderRadius.circular(12), border: Border.all(color: borderCol)),
        child: Row(children: [
          Icon(Icons.show_chart, color: Color(0xFF00d2ff), size: 20),
          SizedBox(width: 8),
          Text('$selectedMarket — ${marketSymbols[selectedMarket]}', style: TextStyle(color: textSec, fontWeight: FontWeight.bold, fontSize: 14)),
          const Spacer(),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
            decoration: BoxDecoration(color: Colors.orange.withOpacity(0.15), borderRadius: BorderRadius.circular(8), border: Border.all(color: Colors.orange.withOpacity(0.3))),
            child: Text('Strategy API Offline', style: TextStyle(color: Colors.orange, fontSize: 11)),
          ),
        ]),
      );
    }

    if (stats == null) return SizedBox.shrink();

    return Row(children: [
      _buildStatCard('STRATEGY RETURN', '${stats!['base_return_pct']?.toStringAsFixed(2)}%', Color(0xFF26a69a), Icons.trending_up),
      _buildStatCard('BUY & HOLD', '${stats!['bnh_return_pct']?.toStringAsFixed(2)}%', Colors.grey, Icons.show_chart),
      _buildStatCard('MAX DRAWDOWN', '${stats!['max_drawdown_pct']?.toStringAsFixed(2)}%', Color(0xFFef5350), Icons.trending_down),
      _buildStatCard('WIN RATE', '${stats!['win_rate_pct']?.toStringAsFixed(1)}%', Color(0xFF1565C0), Icons.pie_chart),
    ]);
  }

  Widget _buildStatCard(String title, String value, Color valueColor, IconData icon) {
    return Expanded(
      child: Container(
        margin: const EdgeInsets.symmetric(horizontal: 4),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
        decoration: BoxDecoration(color: cardCol, borderRadius: BorderRadius.circular(12), border: Border.all(color: borderCol)),
        child: Row(children: [
          Icon(icon, color: valueColor.withOpacity(0.6), size: 20),
          SizedBox(width: 10),
          Expanded(
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(title, style: TextStyle(color: Colors.grey, fontSize: 10, letterSpacing: 1, fontWeight: FontWeight.w500)),
              SizedBox(height: 4),
              Text(value, style: TextStyle(color: valueColor, fontSize: 18, fontWeight: FontWeight.bold)),
            ]),
          ),
        ]),
      ),
    );
  }
}
