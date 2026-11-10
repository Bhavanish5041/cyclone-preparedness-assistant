import React, { useState, useMemo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { MapContainer, TileLayer, CircleMarker, Popup, useMap } from 'react-leaflet';
import { Link } from 'react-router-dom';
import {
  ExternalLink,
  Filter,
  Layers,
  MapPin,
  RotateCcw,
  X,
} from 'lucide-react';
import { fetchMapCommunities } from '../services/api';
import { MapCommunityPoint } from '../types';
import { Header } from '../components/Header';
import { Card } from '../components/Card';
import { Badge } from '../components/Badge';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorState } from '../components/ErrorState';
import { BAND_COLORS } from '../utils/constants';
import { formatNumber } from '../utils/formatting';

// Component to dynamically re-center map if district filter changes
const MapRecenter: React.FC<{ center: [number, number]; zoom: number }> = ({ center, zoom }) => {
  const map = useMap();
  React.useEffect(() => {
    map.setView(center, zoom);
  }, [center, zoom, map]);
  return null;
};

export const MapPage: React.FC = () => {
  const [districtFilter, setDistrictFilter] = useState<string>('');
  const [bandFilter, setBandFilter] = useState<string>('');
  const [tierFilter, setTierFilter] = useState<string>('');
  const [selectedPoint, setSelectedPoint] = useState<MapCommunityPoint | null>(null);

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['mapCommunities', districtFilter, bandFilter, tierFilter],
    queryFn: () =>
      fetchMapCommunities({
        district: districtFilter || undefined,
        priority_band: bandFilter || undefined,
        tier: tierFilter || undefined,
      }),
  });

  // Calculate center based on district
  const { center, zoom } = useMemo<{ [key: string]: any }>(() => {
    if (districtFilter === 'Dakshina Kannada') return { center: [12.87, 75.1], zoom: 10 };
    if (districtFilter === 'Udupi') return { center: [13.45, 74.85], zoom: 10 };
    if (districtFilter === 'Uttara Kannada') return { center: [14.65, 74.55], zoom: 9 };
    return { center: [13.85, 74.8], zoom: 8 };
  }, [districtFilter]);

  const points = data?.points || [];

  return (
    <div className="space-y-4">
      <Header
        title="Geospatial Intelligence Map"
        subtitle="Interactive spatial decision-support mapping covering all 1,879 administrative units across coastal Karnataka."
        badge="GIS Studio"
      />

      {/* Filter HUD Bar */}
      <Card className="p-3 sm:p-3.5">
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-1.5 text-xs font-mono uppercase tracking-wider text-taupe mr-1">
            <Filter className="w-3.5 h-3.5 text-bronze" />
            Filters:
          </div>

          {/* District Scope */}
          <select
            value={districtFilter}
            onChange={(e) => setDistrictFilter(e.target.value)}
            className="text-xs bg-roast border border-chocolate rounded-md px-3 py-1.5 text-ivory focus:outline-none focus:border-bronze"
          >
            <option value="">All Districts (1,879 Units)</option>
            <option value="Dakshina Kannada">Dakshina Kannada (369 Units)</option>
            <option value="Udupi">Udupi (253 Units)</option>
            <option value="Uttara Kannada">Uttara Kannada (1,257 Units)</option>
          </select>

          {/* Priority Band Scope */}
          <select
            value={bandFilter}
            onChange={(e) => setBandFilter(e.target.value)}
            className="text-xs bg-roast border border-chocolate rounded-md px-3 py-1.5 text-ivory focus:outline-none focus:border-bronze"
          >
            <option value="">All Priority Bands</option>
            <option value="BAND_1_CRITICAL">Band 1: Critical (155)</option>
            <option value="BAND_2_SPECIALIZED">Band 2: Specialized (829)</option>
            <option value="BAND_3_MODERATE">Band 3: Moderate (299)</option>
            <option value="BAND_4_ADEQUATE">Band 4: Adequate (596)</option>
          </select>

          {/* AHP Tier Scope */}
          <select
            value={tierFilter}
            onChange={(e) => setTierFilter(e.target.value)}
            className="text-xs bg-roast border border-chocolate rounded-md px-3 py-1.5 text-ivory focus:outline-none focus:border-bronze"
          >
            <option value="">All Preparedness Tiers</option>
            <option value="High">High Preparedness (&gt;60)</option>
            <option value="Medium">Medium Preparedness (40–60)</option>
            <option value="Low">Low Preparedness (&lt;40)</option>
          </select>

          {(districtFilter || bandFilter || tierFilter) && (
            <button
              onClick={() => {
                setDistrictFilter('');
                setBandFilter('');
                setTierFilter('');
              }}
              className="inline-flex items-center gap-1 text-xs text-taupe hover:text-ivory px-2 py-1 transition-colors"
            >
              <RotateCcw className="w-3 h-3" />
              Reset Filters
            </button>
          )}

          <div className="ml-auto text-xs font-mono text-taupe">
            Active Units: <span className="font-semibold text-ivory tabular-nums">{points.length}</span> / 1,879
          </div>
        </div>
      </Card>

      {/* Map View & Inspector Container */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 relative">
        <div className="lg:col-span-3 h-[620px] rounded-lg overflow-hidden border border-chocolate relative bg-roast">
          {isLoading && (
            <div className="absolute inset-0 bg-roast/80 backdrop-blur-sm z-30 flex items-center justify-center">
              <LoadingSpinner message="Plotting verified administrative coordinates..." />
            </div>
          )}

          {error && (
            <div className="absolute inset-0 bg-roast/90 z-30 flex items-center justify-center p-6">
              <ErrorState
                title="Failed to load geospatial data"
                message={error instanceof Error ? error.message : 'Unknown map error'}
                onRetry={() => refetch()}
              />
            </div>
          )}

          <MapContainer
            center={center}
            zoom={zoom}
            className="w-full h-full"
            scrollWheelZoom={true}
          >
            <MapRecenter center={center} zoom={zoom} />
            <TileLayer
              attribution='&copy; <a href="https://www.esri.com/">Esri</a>, USGS, NOAA'
              url="https://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}"
            />

            {points.map((pt) => {
              const colorInfo = BAND_COLORS[pt.band] || { hex: '#76503A' };
              const isSelected = selectedPoint?.pc11_id === pt.pc11_id;

              return (
                <CircleMarker
                  key={pt.pc11_id}
                  center={[pt.lat, pt.lon]}
                  radius={isSelected ? 7 : 4}
                  pathOptions={{
                    color: isSelected ? '#ffffff' : colorInfo.hex,
                    fillColor: colorInfo.hex,
                    fillOpacity: isSelected ? 1.0 : 0.8,
                    weight: isSelected ? 2.5 : 1.0,
                  }}
                  eventHandlers={{
                    click: () => {
                      setSelectedPoint(pt);
                    },
                  }}
                >
                  <Popup>
                    <div className="p-1 max-w-[210px] text-xs space-y-1">
                      <div className="font-display font-bold text-ivory text-sm">{pt.name}</div>
                      <div className="text-taupe text-[11px]">{pt.district} • {pt.subdistrict || 'N/A'}</div>
                      <div className="flex gap-1 pt-1">
                        <Badge type="band" value={pt.band} />
                        <Badge type="tier" value={pt.tier} />
                      </div>
                      <div className="text-[11px] text-taupe pt-1">
                        AHP Score: <span className="font-mono font-semibold text-ivory">{pt.score} / 100</span>
                      </div>
                      <div className="text-[11px] text-taupe">
                        Priority Rank: <span className="font-mono font-semibold text-ivory">#{pt.priority_rank}</span>
                      </div>
                    </div>
                  </Popup>
                </CircleMarker>
              );
            })}
          </MapContainer>

          {/* Floating Map Legend */}
          <div className="absolute bottom-4 left-4 z-[400] bg-espresso/95 border border-chocolate rounded-md p-3 shadow-xl backdrop-blur-md text-xs space-y-1.5">
            <div className="font-mono text-taupe text-[10px] uppercase tracking-wider mb-1 flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-bronze" />
              Priority Bands Legend
            </div>
            {Object.entries(BAND_COLORS).map(([bandKey, style]) => (
              <div key={bandKey} className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: style.hex }} />
                <span className="text-ivory/90 text-[11px]">{style.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Selected Community Information Drawer */}
        <div className="h-[620px] overflow-y-auto">
          {selectedPoint ? (
            <Card className="h-full flex flex-col justify-between">
              <div>
                <div className="flex items-start justify-between pb-3 border-b border-chocolate">
                  <div>
                    <h3 className="font-display text-base font-bold text-ivory">{selectedPoint.name}</h3>
                    <p className="text-xs text-taupe mt-0.5">
                      {selectedPoint.district} • {selectedPoint.subdistrict || 'N/A'}
                    </p>
                  </div>
                  <button
                    onClick={() => setSelectedPoint(null)}
                    className="text-taupe hover:text-ivory p-1"
                    aria-label="Close inspector"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>

                <div className="space-y-4 my-4">
                  {/* Badges */}
                  <div className="flex flex-wrap gap-1.5">
                    <Badge type="band" value={selectedPoint.band} />
                    <Badge type="tier" value={selectedPoint.tier} />
                    {selectedPoint.archetype_id !== null && selectedPoint.archetype_id !== undefined && (
                      <Badge type="cluster" value={selectedPoint.archetype_id} />
                    )}
                  </div>

                  {/* Priority & Score Highlight */}
                  <div className="grid grid-cols-2 gap-2 text-center">
                    <div className="p-2.5 rounded bg-roast border border-chocolate">
                      <div className="text-[10px] font-mono text-taupe uppercase tracking-wider">AHP Score</div>
                      <div className="font-display text-lg font-bold font-mono text-[#4EA5D9] mt-0.5 tabular-nums">
                        {selectedPoint.score}
                      </div>
                      <div className="text-[10px] text-taupe">out of 100.0</div>
                    </div>
                    <div className="p-2.5 rounded bg-roast border border-chocolate">
                      <div className="text-[10px] font-mono text-taupe uppercase tracking-wider">Priority Rank</div>
                      <div className="font-display text-lg font-bold font-mono text-[#E05D52] mt-0.5 tabular-nums">
                        #{selectedPoint.priority_rank}
                      </div>
                      <div className="text-[10px] text-taupe">of 1,879 Units</div>
                    </div>
                  </div>

                  {/* Indicators Table */}
                  <div className="space-y-2 text-xs border-t border-chocolate pt-3">
                    <div className="flex justify-between py-1 border-b border-chocolate/50">
                      <span className="text-taupe">Coast Proximity:</span>
                      <span className="font-mono text-ivory tabular-nums">
                        {selectedPoint.dist_coast_km !== null ? `${selectedPoint.dist_coast_km} km` : 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-chocolate/50">
                      <span className="text-taupe">Nearest Shelter:</span>
                      <span className="font-mono text-ivory tabular-nums">
                        {selectedPoint.dist_nearest_shelter_km !== null ? `${selectedPoint.dist_nearest_shelter_km} km` : 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-chocolate/50">
                      <span className="text-taupe">Nearest Hospital:</span>
                      <span className="font-mono text-ivory tabular-nums">
                        {selectedPoint.dist_nearest_hospital_km !== null ? `${selectedPoint.dist_nearest_hospital_km} km` : 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between py-1 border-b border-chocolate/50">
                      <span className="text-taupe">Population:</span>
                      <span className="font-mono text-ivory tabular-nums">
                        {formatNumber(selectedPoint.population)}
                      </span>
                    </div>
                    <div className="flex justify-between py-1">
                      <span className="text-taupe">Census PC11 ID:</span>
                      <span className="font-mono text-taupe">{selectedPoint.pc11_id}</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Action Button */}
              <div className="pt-3 border-t border-chocolate">
                <Link
                  to={`/communities/${selectedPoint.pc11_id}`}
                  className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 text-xs font-semibold rounded bg-bronze hover:bg-bronze/90 text-roast transition-colors"
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                  Inspect Full Dossier
                </Link>
              </div>
            </Card>
          ) : (
            <Card className="h-full flex flex-col items-center justify-center text-center p-6 text-taupe">
              <MapPin className="w-8 h-8 text-walnut mb-2.5" />
              <h4 className="font-display text-sm font-semibold text-ivory">No Unit Selected</h4>
              <p className="text-xs text-taupe max-w-xs mt-1.5 leading-relaxed">
                Click any circle marker on the map to inspect its authoritative AHP score, priority band, and proximity indicators.
              </p>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
};
