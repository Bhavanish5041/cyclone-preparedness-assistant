import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import {
  ChevronLeft,
  ChevronRight,
  Filter,
  Search,
  ArrowUpDown,
  ExternalLink,
  RotateCcw,
} from 'lucide-react';
import { fetchCommunities } from '../services/api';
import { Header } from '../components/Header';
import { Card } from '../components/Card';
import { Badge } from '../components/Badge';
import { LoadingSpinner } from '../components/LoadingSpinner';
import { ErrorState } from '../components/ErrorState';
import { EmptyState } from '../components/EmptyState';
import { formatNumber } from '../utils/formatting';

export const CommunitiesPage: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [districtFilter, setDistrictFilter] = useState('');
  const [tierFilter, setTierFilter] = useState('');
  const [bandFilter, setBandFilter] = useState('');
  const [sortBy, setSortBy] = useState('priority_rank');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('asc');
  const [page, setPage] = useState(1);
  const limit = 20;

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: [
      'communities',
      searchTerm,
      districtFilter,
      tierFilter,
      bandFilter,
      sortBy,
      sortOrder,
      page,
    ],
    queryFn: () =>
      fetchCommunities({
        search: searchTerm || undefined,
        district: districtFilter || undefined,
        tier: tierFilter || undefined,
        priority_band: bandFilter || undefined,
        sort_by: sortBy,
        sort_order: sortOrder,
        page,
        limit,
      }),
  });

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSearchTerm(searchInput);
    setPage(1);
  };

  const toggleSort = (field: string) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(field);
      setSortOrder('asc');
    }
    setPage(1);
  };

  const items = data?.items || [];
  const totalPages = data?.total_pages || 1;
  const total = data?.total || 0;

  return (
    <div className="space-y-4">
      <Header
        title="Community Explorer"
        subtitle="Searchable registry and multi-criteria vulnerability indicators across 1,879 revenue villages and census towns in coastal Karnataka."
        badge="Registry"
      />

      {/* Search & Filter Header Card */}
      <Card className="p-4 space-y-3">
        <form onSubmit={handleSearchSubmit} className="flex gap-2">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-taupe absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              placeholder="Search by community name, Census PC11 ID, or SHRID..."
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              className="w-full text-xs bg-roast border border-chocolate rounded-md pl-9 pr-3 py-2 text-ivory placeholder-taupe/60 focus:outline-none focus:border-bronze"
            />
          </div>
          <button
            type="submit"
            className="px-4 py-2 text-xs font-semibold rounded-md bg-bronze hover:bg-bronze/90 text-roast transition-colors"
          >
            Search
          </button>
          {searchTerm && (
            <button
              type="button"
              onClick={() => {
                setSearchTerm('');
                setSearchInput('');
                setPage(1);
              }}
              className="px-3 py-2 text-xs text-taupe hover:text-ivory border border-chocolate rounded-md transition-colors"
            >
              Clear
            </button>
          )}
        </form>

        <div className="flex flex-wrap items-center gap-3 pt-2.5 border-t border-chocolate">
          <div className="flex items-center gap-1.5 text-xs text-taupe font-mono uppercase tracking-wider">
            <Filter className="w-3.5 h-3.5 text-bronze" />
            Filters:
          </div>

          <select
            value={districtFilter}
            onChange={(e) => {
              setDistrictFilter(e.target.value);
              setPage(1);
            }}
            className="text-xs bg-roast border border-chocolate rounded-md px-2.5 py-1.5 text-ivory focus:outline-none focus:border-bronze"
          >
            <option value="">All Districts</option>
            <option value="Dakshina Kannada">Dakshina Kannada</option>
            <option value="Udupi">Udupi</option>
            <option value="Uttara Kannada">Uttara Kannada</option>
          </select>

          <select
            value={bandFilter}
            onChange={(e) => {
              setBandFilter(e.target.value);
              setPage(1);
            }}
            className="text-xs bg-roast border border-chocolate rounded-md px-2.5 py-1.5 text-ivory focus:outline-none focus:border-bronze"
          >
            <option value="">All Priority Bands</option>
            <option value="BAND_1_CRITICAL">Band 1: Critical Priority</option>
            <option value="BAND_2_SPECIALIZED">Band 2: Specialized Readiness</option>
            <option value="BAND_3_MODERATE">Band 3: Moderate Priority</option>
            <option value="BAND_4_ADEQUATE">Band 4: Adequate Baseline</option>
          </select>

          <select
            value={tierFilter}
            onChange={(e) => {
              setTierFilter(e.target.value);
              setPage(1);
            }}
            className="text-xs bg-roast border border-chocolate rounded-md px-2.5 py-1.5 text-ivory focus:outline-none focus:border-bronze"
          >
            <option value="">All AHP Tiers</option>
            <option value="High">High (&gt;60)</option>
            <option value="Medium">Medium (40–60)</option>
            <option value="Low">Low (&lt;40)</option>
          </select>

          {(districtFilter || bandFilter || tierFilter || searchTerm) && (
            <button
              onClick={() => {
                setDistrictFilter('');
                setBandFilter('');
                setTierFilter('');
                setSearchTerm('');
                setSearchInput('');
                setPage(1);
              }}
              className="inline-flex items-center gap-1 text-xs text-taupe hover:text-ivory px-2 py-1 transition-colors"
            >
              <RotateCcw className="w-3 h-3" />
              Reset Filters
            </button>
          )}

          <div className="ml-auto text-xs font-mono text-taupe">
            Total Matching: <span className="font-semibold text-ivory tabular-nums">{formatNumber(total)}</span>
          </div>
        </div>
      </Card>

      {/* Main Table Card */}
      <Card className="p-0 overflow-hidden">
        {isLoading ? (
          <div className="py-20">
            <LoadingSpinner message="Querying administrative unit records..." />
          </div>
        ) : error ? (
          <div className="p-8">
            <ErrorState
              title="Error loading community table"
              message={error instanceof Error ? error.message : 'Unknown query error'}
              onRetry={() => refetch()}
            />
          </div>
        ) : items.length === 0 ? (
          <EmptyState
            title="No communities match your criteria"
            description="Try modifying search query, district filter, or priority band restrictions."
          />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="bg-roast border-b border-chocolate text-taupe font-mono uppercase tracking-wider text-[10px]">
                  <th
                    className="p-3.5 cursor-pointer hover:text-ivory transition-colors"
                    onClick={() => toggleSort('village_name')}
                  >
                    <div className="flex items-center gap-1">
                      <span>Community Name</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th className="p-3.5">District & Taluk</th>
                  <th
                    className="p-3.5 cursor-pointer hover:text-ivory transition-colors"
                    onClick={() => toggleSort('preparedness_score')}
                  >
                    <div className="flex items-center gap-1">
                      <span>AHP Score</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th className="p-3.5">Preparedness Tier</th>
                  <th className="p-3.5">Priority Band</th>
                  <th
                    className="p-3.5 cursor-pointer hover:text-ivory transition-colors text-right"
                    onClick={() => toggleSort('priority_rank')}
                  >
                    <div className="flex items-center justify-end gap-1">
                      <span>Priority Rank</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th
                    className="p-3.5 cursor-pointer hover:text-ivory transition-colors text-right"
                    onClick={() => toggleSort('dist_coast_km')}
                  >
                    <div className="flex items-center justify-end gap-1">
                      <span>Coast Dist</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th
                    className="p-3.5 cursor-pointer hover:text-ivory transition-colors text-right"
                    onClick={() => toggleSort('population')}
                  >
                    <div className="flex items-center justify-end gap-1">
                      <span>Population</span>
                      <ArrowUpDown className="w-3 h-3" />
                    </div>
                  </th>
                  <th className="p-3.5 text-center">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-chocolate/60">
                {items.map((c) => (
                  <tr
                    key={c.pc11_id}
                    className="hover:bg-chocolate/20 transition-colors group"
                  >
                    <td className="p-3.5">
                      <Link
                        to={`/communities/${c.pc11_id}`}
                        className="font-medium text-ivory group-hover:text-bronze transition-colors"
                      >
                        {c.village_name}
                      </Link>
                      <div className="text-[10px] font-mono text-taupe">
                        PC11: {c.pc11_id}
                      </div>
                    </td>
                    <td className="p-3.5">
                      <div className="text-ivory/90">{c.district_name}</div>
                      <div className="text-[11px] text-taupe">
                        Taluk: {c.subdistrict_name || 'N/A'}
                      </div>
                    </td>
                    <td className="p-3.5 font-mono font-semibold text-ivory tabular-nums">
                      {c.preparedness_score}
                    </td>
                    <td className="p-3.5">
                      <Badge type="tier" value={c.preparedness_tier} />
                    </td>
                    <td className="p-3.5">
                      <Badge type="band" value={c.priority_band} />
                    </td>
                    <td className="p-3.5 text-right font-mono font-semibold text-[#E05D52] tabular-nums">
                      #{c.priority_rank}
                    </td>
                    <td className="p-3.5 text-right font-mono text-ivory/80 tabular-nums">
                      {c.dist_coast_km !== null ? `${c.dist_coast_km} km` : 'N/A'}
                    </td>
                    <td className="p-3.5 text-right font-mono text-ivory/80 tabular-nums">
                      {formatNumber(c.population)}
                    </td>
                    <td className="p-3.5 text-center">
                      <Link
                        to={`/communities/${c.pc11_id}`}
                        className="inline-flex items-center gap-1 text-[11px] px-2.5 py-1 rounded bg-chocolate hover:bg-bronze text-taupe hover:text-roast transition-colors"
                      >
                        <ExternalLink className="w-3 h-3" />
                        Dossier
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination Controls */}
        <div className="p-3.5 border-t border-chocolate bg-roast/60 flex items-center justify-between text-xs text-taupe">
          <div>
            Showing Page <span className="font-semibold text-ivory tabular-nums">{page}</span> of{' '}
            <span className="font-semibold text-ivory tabular-nums">{totalPages}</span>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page <= 1}
              className="inline-flex items-center gap-1 px-3 py-1.5 rounded border border-chocolate bg-espresso text-ivory disabled:opacity-30 disabled:cursor-not-allowed hover:bg-chocolate transition-colors"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
              Previous
            </button>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page >= totalPages}
              className="inline-flex items-center gap-1 px-3 py-1.5 rounded border border-chocolate bg-espresso text-ivory disabled:opacity-30 disabled:cursor-not-allowed hover:bg-chocolate transition-colors"
            >
              Next
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </Card>
    </div>
  );
};
